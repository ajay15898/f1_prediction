import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import fastf1

# Ensure FastF1 caching is enabled
CACHE_PATH = Path("f1cache")
CACHE_PATH.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_PATH))

OPENF1_BASE = "https://api.openf1.org/v1"
TEAM_NAME = "Ferrari"
DEFAULT_SEASON = 2025


st.set_page_config(
    page_title="Ferrari Performance Dashboard",
    page_icon="🏎️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def fetch_openf1(endpoint: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """Fetch data from the OpenF1 API and convert to a DataFrame."""
    if params is None:
        params = {}
    url = f"{OPENF1_BASE}/{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to reach OpenF1 endpoint '{endpoint}': {exc}")
        return pd.DataFrame()


def normalize_race_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


@st.cache_data(show_spinner=False)
def get_team_seasons(
    team: str,
    start_year: int = 2018,
    end_year: Optional[int] = None,
) -> List[int]:
    """Discover seasons with available race results for the given team."""
    if end_year is None:
        end_year = max(DEFAULT_SEASON, pd.Timestamp.utcnow().year)

    df = fetch_openf1(
        "results",
        {
            "team_name": team,
            "session_name": "Race",
            "date_start": f"{start_year}-01-01",
            "date_end": f"{end_year}-12-31",
            "limit": 2000,
        },
    )
    if df.empty or "season" not in df.columns:
        return sorted({DEFAULT_SEASON})

    seasons = {
        int(season)
        for season in pd.to_numeric(df["season"], errors="coerce").dropna().astype(int)
    }
    seasons.add(DEFAULT_SEASON)
    return sorted(seasons)


@st.cache_data(show_spinner=False)
def find_latest_season_with_results(
    requested_season: int,
    team: str,
    seasons: Optional[Iterable[int]] = None,
) -> Tuple[int, pd.DataFrame]:
    """Return the most recent season (<= requested) that has available results."""

    season_pool = sorted({*seasons} if seasons else set())
    if not season_pool:
        season_pool = list(range(requested_season, 2017, -1))

    min_year = min(season_pool) if season_pool else 2018
    search_years: List[int] = []

    for year in range(requested_season, min_year - 1, -1):
        search_years.append(year)

    for year in sorted(season_pool, reverse=True):
        if year > requested_season:
            search_years.append(year)
        elif year not in search_years:
            search_years.append(year)

    seen: set[int] = set()
    ordered_years: List[int] = []
    for year in search_years:
        if year not in seen:
            ordered_years.append(year)
            seen.add(year)

    for year in ordered_years:
        results = get_team_results(year, team)
        if not results.empty:
            return year, results

    return requested_season, pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_team_results(season: int, team: str) -> pd.DataFrame:
    df = fetch_openf1(
        "results",
        {
            "season": season,
            "team_name": team,
            "session_name": "Race",
        },
    )
    if df.empty:
        return df

    # Normalize useful columns
    rename_map = {
        "meeting_name": "race_name",
        "grand_prix": "race_name",
        "session_name": "session_name",
        "date_start": "session_start",
        "date": "session_start",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "race_name" not in df.columns:
        if "session_name" in df.columns:
            df["race_name"] = df["session_name"]
        else:
            df["race_name"] = "Race"

    if "session_start" in df.columns:
        df["session_start"] = pd.to_datetime(df["session_start"], errors="coerce")
        df = df.sort_values("session_start")
    elif "round" in df.columns:
        df = df.sort_values("round")

    if "round" not in df.columns:
        df["round"] = (
            df.groupby("race_name").ngroup() + 1
        )

    df["race_key"] = df.get("meeting_key", df.get("session_key", df["round"]))
    df["race_label"] = df["race_name"].fillna("Race")
    df["race_slug"] = df["race_label"].map(normalize_race_name)

    for column in ["grid_position", "position", "points", "laps"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "status" not in df.columns:
        df["status"] = np.where(df["laps"].notnull(), "Finished", "Unknown")

    return df


@st.cache_data(show_spinner=False)
def get_team_pitstops(season: int, team: str) -> pd.DataFrame:
    df = fetch_openf1(
        "pit",
        {
            "season": season,
            "team_name": team,
        },
    )
    if df.empty:
        return df

    rename_map = {
        "meeting_name": "race_name",
        "grand_prix": "race_name",
        "duration": "pit_duration",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "pit_duration" in df.columns:
        df["pit_duration"] = pd.to_numeric(df["pit_duration"], errors="coerce")

    df["race_slug"] = df.get("race_name", df.get("meeting_name", "Race")).map(
        normalize_race_name
    )
    if "round" not in df.columns and "session_key" in df.columns:
        df["round"] = df.groupby("session_key").ngroup() + 1
    return df


def filter_results(
    df: pd.DataFrame, drivers: Iterable[str], races: Iterable[str]
) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    if drivers:
        driver_cols = [col for col in ["driver_number", "driver_code", "driver_name"] if col in df.columns]
        if driver_cols:
            driver_mask = False
            for col in driver_cols:
                driver_mask = driver_mask | filtered[col].astype(str).isin(drivers)
            filtered = filtered[driver_mask]
    if races:
        filtered = filtered[filtered["race_slug"].isin([normalize_race_name(r) for r in races])]
    return filtered


def build_points_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    if df.empty or "points" not in df.columns:
        return None
    race_order = df[["race_slug", "race_label", "round"]].drop_duplicates().sort_values("round")
    points_by_race = (
        df.groupby(["race_slug", "race_label", "round"], as_index=False)["points"].sum()
        .sort_values("round")
    )
    points_by_race["cumulative_points"] = points_by_race["points"].cumsum()

    fig = go.Figure()
    fig.add_bar(
        x=points_by_race["race_label"],
        y=points_by_race["points"],
        name="Points",
        marker_color="#e10600",
    )
    fig.add_trace(
        go.Scatter(
            x=points_by_race["race_label"],
            y=points_by_race["cumulative_points"],
            mode="lines+markers",
            name="Cumulative",
            marker=dict(color="#1f77b4"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Points by Race",
        yaxis=dict(title="Points"),
        yaxis2=dict(title="Cumulative Points", overlaying="y", side="right"),
        bargap=0.2,
        hovermode="x unified",
    )
    return fig


def build_grid_finish_figures(df: pd.DataFrame) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    if df.empty or "grid_position" not in df.columns or "position" not in df.columns:
        return None, None
    valid = df.dropna(subset=["grid_position", "position"])
    if valid.empty:
        return None, None
    valid["delta"] = valid["grid_position"] - valid["position"]
    scatter = px.scatter(
        valid,
        x="grid_position",
        y="position",
        color="driver_code" if "driver_code" in valid.columns else "driver_name",
        hover_data=["race_label", "delta"],
        title="Grid vs Finish Position",
        labels={"grid_position": "Grid", "position": "Finish"},
    )
    scatter.update_traces(marker=dict(size=12, line=dict(width=1, color="#ffffff")))
    scatter.update_layout(yaxis=dict(autorange="reversed"))

    distribution = px.histogram(
        valid,
        x="delta",
        nbins=15,
        color="driver_code" if "driver_code" in valid.columns else None,
        title="Grid to Finish Delta Distribution",
        labels={"delta": "Grid - Finish"},
    )
    distribution.update_layout(bargap=0.1)
    return scatter, distribution


def build_dnf_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    statuses = df[~df["status"].str.contains("finish", case=False, na=False)]
    if statuses.empty:
        return pd.DataFrame()
    columns = [
        col
        for col in [
            "race_label",
            "driver_name",
            "driver_code",
            "status",
            "laps",
        ]
        if col in statuses.columns
    ]
    return statuses[columns].rename(columns={"laps": "Lap Count"})


def build_pit_time_chart(pits: pd.DataFrame, results: pd.DataFrame) -> Optional[go.Figure]:
    if pits.empty or "pit_duration" not in pits.columns:
        return None
    if "race_label" not in pits.columns:
        mapping = results[["race_slug", "race_label"]].drop_duplicates()
        pits = pits.merge(mapping, how="left", left_on="race_slug", right_on="race_slug")
    pit_summary = (
        pits.dropna(subset=["pit_duration"])
        .groupby(["race_slug", "race_label"], as_index=False)["pit_duration"]
        .sum()
    )
    if pit_summary.empty:
        return None
    fig = px.bar(
        pit_summary.sort_values("race_label"),
        x="race_label",
        y="pit_duration",
        title="Pit Stop Time Loss per Race",
        labels={"pit_duration": "Total Pit Duration (s)", "race_label": "Race"},
        color_discrete_sequence=["#ff5f5f"],
    )
    return fig


def _sc_mask(track_status: str) -> bool:
    if not isinstance(track_status, str):
        return False
    sc_codes = {"4", "5", "6", "7"}
    return any(code in track_status.split() for code in sc_codes)


@st.cache_data(show_spinner=False)
def get_fastf1_metrics(
    season: int,
    races: pd.DataFrame,
    driver_codes: Iterable[str],
) -> pd.DataFrame:
    if races.empty:
        return pd.DataFrame()
    driver_codes = list(driver_codes)
    if not driver_codes:
        return pd.DataFrame()

    summaries: List[Dict] = []
    unique_races = (
        races[["round", "race_label", "race_slug"]]
        .drop_duplicates()
        .sort_values("round")
    )
    for _, race in unique_races.iterrows():
        round_number = race["round"]
        race_label = race["race_label"]
        race_slug = race["race_slug"]
        try:
            event = fastf1.get_event(season, int(round_number))
            session = event.get_session("R")
            session.load(laps=True, telemetry=False, weather=False)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"FastF1 data unavailable for round {round_number}: {exc}")
            continue

        laps = session.laps
        if laps.empty:
            continue
        field_quick = laps.pick_quicklaps()
        if field_quick.empty:
            continue
        field_quick["LapTimeSeconds"] = field_quick["LapTime"].dt.total_seconds()
        stint_medians = (
            field_quick.groupby("Stint")["LapTimeSeconds"].median().dropna()
        )

        for code in driver_codes:
            driver_laps = field_quick[field_quick["Driver"] == code]
            if driver_laps.empty:
                continue
            driver_laps = driver_laps.copy()
            driver_laps["LapTimeSeconds"] = driver_laps["LapTime"].dt.total_seconds()
            stint_group = driver_laps.groupby("Stint")
            for stint, stint_data in stint_group:
                median = stint_medians.get(stint)
                if pd.isna(median):
                    continue
                delta = stint_data["LapTimeSeconds"].mean() - median
                summaries.append(
                    {
                        "round": round_number,
                        "race_label": race_label,
                        "race_slug": race_slug,
                        "driver_code": code,
                        "stint": stint,
                        "avg_lap": stint_data["LapTimeSeconds"].mean(),
                        "median_lap": median,
                        "stint_pace_delta": delta,
                    }
                )

            driver_all_laps = laps[laps["Driver"] == code]
            sc_laps = driver_all_laps[driver_all_laps["TrackStatus"].apply(_sc_mask)]
            sc_count = sc_laps["LapNumber"].nunique()
            summaries.append(
                {
                    "round": round_number,
                    "race_label": race_label,
                    "race_slug": race_slug,
                    "driver_code": code,
                    "metric": "sc_laps",
                    "value": sc_count,
                }
            )
    if not summaries:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summaries)
    return summary_df


def build_stint_pace_views(metrics: pd.DataFrame) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    if metrics.empty:
        return None, None

    stint_metrics = metrics[metrics["stint"].notna()]
    if stint_metrics.empty:
        return None, None

    per_race = (
        stint_metrics.groupby(["race_label", "race_slug"])["stint_pace_delta"].mean().reset_index()
    )
    per_race_fig = px.bar(
        per_race,
        x="race_label",
        y="stint_pace_delta",
        title="Average Stint Pace Delta vs Field Median (per race)",
        labels={"stint_pace_delta": "Delta (s)", "race_label": "Race"},
        color_discrete_sequence=["#c00000"],
    )

    per_driver = (
        stint_metrics.groupby(["driver_code"])["stint_pace_delta"].mean().reset_index()
    )
    per_driver_fig = px.bar(
        per_driver,
        x="driver_code",
        y="stint_pace_delta",
        title="Average Stint Pace Delta vs Field Median (per driver)",
        labels={"stint_pace_delta": "Delta (s)", "driver_code": "Driver"},
        color_discrete_sequence=["#ff8c69"],
    )
    return per_race_fig, per_driver_fig


def build_sc_exposure_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    sc = metrics[metrics.get("metric") == "sc_laps"]
    if sc.empty:
        return pd.DataFrame()
    pivot = sc.pivot_table(
        index=["race_label"],
        columns="driver_code",
        values="value",
        aggfunc="sum",
    ).fillna(0)
    pivot = pivot.astype(int)
    pivot.columns.name = None
    pivot = pivot.reset_index().rename(columns={"race_label": "Race"})
    return pivot


def summarize_driver_features(
    results: pd.DataFrame,
    pits: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    race_driver = results[[
        "race_label",
        "race_slug",
        "driver_code",
        "driver_name",
        "grid_position",
        "position",
        "points",
        "status",
    ]].copy()

    race_driver["finish_delta"] = race_driver["grid_position"] - race_driver["position"]

    pit_summary = pd.DataFrame()
    if not pits.empty and "pit_duration" in pits.columns:
        group_cols = ["race_slug"]
        if "driver_number" in pits.columns:
            group_cols.append("driver_number")
        elif "driver_code" in pits.columns:
            group_cols.append("driver_code")
        elif "driver_name" in pits.columns:
            group_cols.append("driver_name")
        pit_summary = (
            pits.dropna(subset=["pit_duration"])
            .groupby(group_cols, as_index=False)["pit_duration"].sum()
        )
        if "driver_number" in pit_summary.columns and "driver_code" in results.columns:
            driver_map = (
                results[["driver_code", "driver_number"]]
                .drop_duplicates()
                .dropna()
            )
            if not driver_map.empty:
                pit_summary = pit_summary.merge(driver_map, on="driver_number", how="left")
        if "driver_name" in pits.columns and "driver_name" not in pit_summary.columns:
            pit_summary = pit_summary.merge(
                pits[["race_slug", "driver_name"]].drop_duplicates(),
                on="race_slug",
                how="left",
            )

    stint_summary = metrics[metrics.get("stint").notna()] if not metrics.empty else pd.DataFrame()
    if not stint_summary.empty:
        stint_summary = (
            stint_summary.groupby(["race_slug", "driver_code"], as_index=False)["stint_pace_delta"].mean()
        )

    sc_summary = build_sc_exposure_table(metrics)
    if not sc_summary.empty:
        sc_summary["race_slug"] = sc_summary["Race"].map(normalize_race_name)

    summary = race_driver
    if not pit_summary.empty:
        pit_summary = pit_summary.rename(columns={"pit_duration": "pit_time_loss"})
        merge_cols = ["race_slug"]
        if "driver_code" in summary.columns and "driver_code" in pit_summary.columns:
            merge_cols.append("driver_code")
        elif "driver_name" in summary.columns and "driver_name" in pit_summary.columns:
            merge_cols.append("driver_name")
        summary = summary.merge(
            pit_summary,
            on=merge_cols,
            how="left",
        )
    if not stint_summary.empty:
        summary = summary.merge(
            stint_summary,
            on=["race_slug", "driver_code"],
            how="left",
        )
    if not sc_summary.empty:
        summary = summary.merge(
            sc_summary.melt(
                id_vars=["Race", "race_slug"], var_name="driver_code", value_name="sc_laps"
            ),
            on=["race_slug", "driver_code"],
            how="left",
        )
    summary["dnf_flag"] = (~summary["status"].str.contains("finish", case=False, na=False)).astype(int)
    return summary


def build_feature_importance_table(summary: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[go.Figure]]:
    if summary.empty:
        return pd.DataFrame(), None
    feature_cols = [
        col
        for col in [
            "grid_position",
            "finish_delta",
            "stint_pace_delta",
            "pit_time_loss",
            "sc_laps",
            "dnf_flag",
        ]
        if col in summary.columns
    ]
    if not feature_cols or "points" not in summary.columns:
        return pd.DataFrame(), None

    model_data = summary.dropna(subset=feature_cols + ["points"])
    if len(model_data) < 6:
        return model_data, None

    X = model_data[feature_cols]
    y = model_data["points"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=20, random_state=42)
    importance = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": perm.importances_mean,
                "std": perm.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
    )
    fig = px.bar(
        importance,
        x="feature",
        y="importance",
        error_y="std",
        title="Feature Importance (Permutation)",
        labels={"feature": "Feature", "importance": "Importance"},
        color_discrete_sequence=["#1f77b4"],
    )
    return model_data, fig


def render_driver_summary(summary: pd.DataFrame, drivers: List[str]) -> None:
    if summary.empty:
        st.info("No driver summary available for the current selection.")
        return

    cols = st.columns(len(drivers) if drivers else 1)
    for idx, driver_code in enumerate(drivers or summary["driver_code"].unique()):
        driver_rows = summary[summary["driver_code"] == driver_code]
        if driver_rows.empty:
            continue
        driver_name = driver_rows["driver_name"].iloc[0] if "driver_name" in driver_rows.columns else driver_code
        avg_quali = driver_rows["grid_position"].mean()
        avg_delta = driver_rows.get("stint_pace_delta", pd.Series(dtype=float)).mean()
        avg_pit = driver_rows.get("pit_time_loss", pd.Series(dtype=float)).mean()
        avg_sc = driver_rows.get("sc_laps", pd.Series(dtype=float)).mean()
        dnf_rate = driver_rows["dnf_flag"].mean()

        with cols[min(idx, len(cols) - 1)]:
            st.subheader(driver_name)
            st.metric("Average Grid Position", f"{avg_quali:.1f}")
            if not np.isnan(avg_delta):
                st.metric("Avg Stint Pace Δ (s)", f"{avg_delta:.3f}")
            if not np.isnan(avg_pit):
                st.metric("Avg Pit Time Loss (s)", f"{avg_pit:.2f}")
            if not np.isnan(avg_sc):
                st.metric("Avg SC Laps", f"{avg_sc:.1f}")
            st.metric("DNF Rate", f"{dnf_rate * 100:.1f}%")


def build_driver_race_filters(results: pd.DataFrame) -> Tuple[List[str], List[str]]:
    driver_options: List[str] = []
    format_func = None
    if not results.empty:
        if {"driver_code", "driver_name"}.issubset(results.columns):
            driver_map = (
                results[["driver_code", "driver_name"]]
                .dropna()
                .drop_duplicates()
                .set_index("driver_code")["driver_name"]
                .to_dict()
            )
            driver_options = sorted(driver_map.keys())
            format_func = lambda code: f"{code} — {driver_map.get(code, code)}"
        elif "driver_code" in results.columns:
            driver_options = sorted(results["driver_code"].dropna().unique().tolist())
        elif "driver_name" in results.columns:
            driver_options = sorted(results["driver_name"].dropna().unique().tolist())

    selected_drivers = st.sidebar.multiselect(
        "Drivers",
        options=driver_options,
        default=driver_options,
        format_func=format_func,
        key="driver_filters",
    )

    race_options: List[str] = []
    if not results.empty:
        race_options = (
            results.sort_values("round")["race_label"].dropna().unique().tolist()
        )

    selected_races = st.sidebar.multiselect(
        "Races",
        options=race_options,
        default=race_options,
        key="race_filters",
    )

    if not selected_drivers:
        selected_drivers = driver_options
    if not selected_races:
        selected_races = race_options
    return selected_drivers, selected_races


def main() -> None:
    st.title(f"{TEAM_NAME} Performance Dashboard")
    st.caption(
        "OpenF1 results combined with FastF1 telemetry to unpack Ferrari's race weekends."
    )

    available_seasons = get_team_seasons(TEAM_NAME)
    if not available_seasons:
        available_seasons = [DEFAULT_SEASON]

    season_options = sorted(set(available_seasons))
    if DEFAULT_SEASON not in season_options:
        season_options.append(DEFAULT_SEASON)
    season_options = sorted(season_options)

    default_index = (
        season_options.index(DEFAULT_SEASON)
        if DEFAULT_SEASON in season_options
        else len(season_options) - 1
    )

    st.sidebar.header("Filters")
    requested_season = st.sidebar.selectbox(
        "Season",
        options=season_options,
        index=default_index,
        key="season_selection",
    )

    active_season, results = find_latest_season_with_results(
        requested_season,
        TEAM_NAME,
        season_options,
    )
    if results.empty:
        st.error("No OpenF1 results available for the selected season.")
        st.stop()

    if active_season != requested_season:
        st.info(
            f"No OpenF1 results available for {requested_season}. Displaying {active_season} instead."
        )

    st.markdown(f"### Season {active_season}")

    driver_filters, race_filters = build_driver_race_filters(results)

    drivers = driver_filters or (
        results["driver_code"].dropna().unique().tolist()
        if "driver_code" in results.columns
        else []
    )
    races = race_filters or results["race_label"].dropna().unique().tolist()

    filtered_results = filter_results(results, drivers, races)
    pitstops = get_team_pitstops(active_season, TEAM_NAME)
    if not pitstops.empty:
        pitstops = pitstops[pitstops["race_slug"].isin(filtered_results["race_slug"].unique())]

    fastf1_metrics = get_fastf1_metrics(
        active_season,
        filtered_results,
        drivers,
    )

    overview_tab, pace_tab, explain_tab = st.tabs(
        [
            "Overview",
            "Pace & Strategy",
            "Explain Performance",
        ]
    )

    with overview_tab:
        st.subheader("Points Progression")
        fig_points = build_points_chart(filtered_results)
        if fig_points:
            st.plotly_chart(fig_points, use_container_width=True)
        else:
            st.info("Points data not available.")

        st.subheader("Qualifying vs Race")
        scatter, distribution = build_grid_finish_figures(filtered_results)
        col1, col2 = st.columns(2)
        with col1:
            if scatter:
                st.plotly_chart(scatter, use_container_width=True)
            else:
                st.info("Grid/finish comparison not available.")
        with col2:
            if distribution:
                st.plotly_chart(distribution, use_container_width=True)
            else:
                st.info("Delta distribution unavailable.")

        st.subheader("Retirements")
        dnf_table = build_dnf_table(filtered_results)
        if not dnf_table.empty:
            st.dataframe(dnf_table, use_container_width=True)
        else:
            st.info("No retirements recorded in the current selection.")

        st.subheader("Pit Stop Efficiency")
        pit_fig = build_pit_time_chart(pitstops, filtered_results)
        if pit_fig:
            st.plotly_chart(pit_fig, use_container_width=True)
        else:
            st.info("Pit stop information unavailable.")

    with pace_tab:
        st.subheader("Stint Pace vs Field")
        race_fig, driver_fig = build_stint_pace_views(fastf1_metrics)
        if race_fig:
            st.plotly_chart(race_fig, use_container_width=True)
        else:
            st.info("Stint pace data unavailable.")
        if driver_fig:
            st.plotly_chart(driver_fig, use_container_width=True)

        st.subheader("Safety Car Exposure")
        sc_table = build_sc_exposure_table(fastf1_metrics)
        if not sc_table.empty:
            st.dataframe(sc_table, use_container_width=True)
        else:
            st.info("No safety car exposure data available.")

    with explain_tab:
        st.subheader("Driver Weekend Summary")
        driver_summary = summarize_driver_features(
            filtered_results,
            pitstops,
            fastf1_metrics,
        )
        render_driver_summary(driver_summary, drivers)

        st.subheader("Feature Importance")
        model_data, feature_fig = build_feature_importance_table(driver_summary)
        if feature_fig:
            st.plotly_chart(feature_fig, use_container_width=True)
        else:
            st.info("Not enough data to compute feature importance.")

        if not model_data.empty:
            st.markdown(
                """
                The Random Forest regressor uses per-race metrics to explain how each
                factor contributes to Ferrari's points haul. Higher importance values
                indicate features that most influence the predicted points when shuffled.
                """
            )

            st.dataframe(
                model_data[
                    [
                        col
                        for col in [
                            "race_label",
                            "driver_name",
                            "grid_position",
                            "finish_delta",
                            "stint_pace_delta",
                            "pit_time_loss",
                            "sc_laps",
                            "dnf_flag",
                            "points",
                        ]
                        if col in model_data.columns
                    ]
                ].rename(
                    columns={
                        "grid_position": "Grid",
                        "finish_delta": "Grid-Finish Δ",
                        "stint_pace_delta": "Stint Pace Δ",
                        "pit_time_loss": "Pit Loss (s)",
                        "sc_laps": "SC Laps",
                        "dnf_flag": "DNF",
                    }
                ),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
