# Ferrari Performance Dashboard

This Streamlit application analyzes the current Formula 1 season for Ferrari by combining race metadata from [OpenF1](https://api.openf1.org) with telemetry and lap data from [FastF1](https://theoehrly.github.io/Fast-F1/). The dashboard summarizes each race weekend through points breakdowns, qualifying versus race performance, pit stop execution, stint pace relative to the field, safety-car exposure, and a lightweight feature-importance view that explains the outcomes.

## Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the dashboard:

   ```bash
   make run
   ```

FastF1 telemetry is cached to `./f1cache` on first run to speed up subsequent loads. By default the dashboard opens on the 2025 Formula 1 season, but you can switch to any supported season, driver, or race via the sidebar filters.
