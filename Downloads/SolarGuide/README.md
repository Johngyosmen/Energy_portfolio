# Solar Monitoring & Predictive Maintenance Dashboard

A Streamlit web application for monitoring solar-plant generation data and identifying records that may need operational attention. It combines generation measurements with weather readings, estimates expected AC power with a Random Forest regression model, flags unusual records with Isolation Forest, and presents health/risk summaries in interactive charts and tables.

> **Important:** The health and failure-risk values in this project are heuristic indicators for exploration. They are not a guarantee that equipment will fail, and they should not be used as the sole basis for safety-critical maintenance decisions.

## Contents

- [Features](#features)
- [Dashboard pages](#dashboard-pages)
- [How the application works](#how-the-application-works)
- [Project files and data requirements](#project-files-and-data-requirements)
- [Requirements](#requirements)
- [Installation and setup](#installation-and-setup)
- [Run the dashboard](#run-the-dashboard)
- [Understanding the main calculations](#understanding-the-main-calculations)
- [Troubleshooting](#troubleshooting)
- [Limitations and suggested improvements](#limitations-and-suggested-improvements)

## Features

- **Plant overview:** summary metrics, a plant-health estimate, anomaly count, and a time-series chart of AC power.
- **Inverter health:** compares average AC power by inverter and identifies the highest and lowest average output in the data.
- **AI power predictions:** compares model-estimated power with actual power and displays the model's feature-importance estimates.
- **Maintenance alerts:** lists records that have a non-normal power-loss status or have been flagged as anomalous.
- **Asset health:** ranks inverters using average health scores and highlights lower-scoring assets.
- **Failure-risk forecast:** ranks inverters by a rule-based failure-risk indicator and displays suggested actions.
- **Interactive visualizations:** charts are created with Plotly and rendered in Streamlit.
- **Cached data loading:** Streamlit caches the CSV-loading function to reduce repeated file reads during normal interactions.

## Dashboard pages

Choose a page from the **Select Page** menu in the sidebar.

| Page | What it shows |
|---|---|
| Plant Overview | Plant health, number of monitored inverters, anomaly count, high-risk record count, average risk, power metrics, and a power-over-time chart. |
| Inverter Health | Average AC power for each inverter, with the highest and lowest average output highlighted. |
| AI Predictions | A scatter plot of expected versus actual AC power and a feature-importance chart. |
| Maintenance Alerts | Records with warning/critical power loss or an anomaly flag, plus a maintenance recommendation and action. |
| Asset Health | Inverter health ranking, critical assets, and a summary of health and failure-risk scores. |
| Failure Forecast | Failure-risk ranking, risk labels, highest risk score, and a table of high-risk records. |

## How the application works

The main processing pipeline is:

1. **Load the source files.** Read the plant generation and weather sensor CSV files with pandas.
2. **Parse timestamps.** Convert each file's `DATE_TIME` column to datetime values.
3. **Combine the datasets.** Perform an inner join on `DATE_TIME` and `PLANT_ID`, keeping rows with matching values in both files.
4. **Remove incomplete model rows.** Exclude records that are missing one or more model features or the `AC_POWER` target.
5. **Train the power model.** Train a `RandomForestRegressor` using irradiation, ambient temperature, and module temperature to estimate AC power.
6. **Estimate expected power.** Use the trained model to calculate an expected AC-power value for each cleaned record.
7. **Calculate power loss.** Compare expected power with measured AC power to calculate a percentage difference.
8. **Detect unusual records.** Use an `IsolationForest` model on AC power, DC power, irradiation, and module temperature.
9. **Calculate indicators.** Derive a heuristic risk score, health score, failure-risk indicator, severity label, and recommended action.
10. **Display the results.** Present metrics, tables, and Plotly charts on the selected dashboard page.

## Project files and data requirements

Keep these files in the same folder:

```text
your-project/
├── streamlit_app.py
├── Plant_2_Generation_Data.csv
├── Plant_2_Weather_Sensor_Data.csv
└── README.md
```

If you use the commented version supplied with this README, it may be named `streamlit_app_commented.py`; use that filename in the run command or rename it to `streamlit_app.py`.

### 1. `Plant_2_Generation_Data.csv`

The application expects generation data containing at least these columns:

- `DATE_TIME` — date and time of the measurement
- `PLANT_ID` — plant identifier
- `SOURCE_KEY` — inverter or source identifier
- `AC_POWER` — measured AC power
- `DC_POWER` — measured DC power

### 2. `Plant_2_Weather_Sensor_Data.csv`

The application expects weather data containing at least:

- `DATE_TIME` — date and time of the weather reading
- `PLANT_ID` — plant identifier
- `IRRADIATION` — solar irradiation measurement
- `AMBIENT_TEMPERATURE` — ambient temperature
- `MODULE_TEMPERATURE` — solar-module temperature

The join produces overlapping source columns with pandas suffixes where needed. The code refers to `SOURCE_KEY_x` for the generation-side source identifier, so verify that this column exists after merging. If your CSV files have different column names, update the code to match the actual headers.

**Data privacy:** use datasets you are authorized to process. Do not commit confidential operational data or credentials to a public repository.

## Requirements

- Python 3.9 or a compatible Python version supported by your installed packages
- Streamlit
- pandas
- NumPy
- scikit-learn
- Plotly
- The two required CSV data files

## Installation and setup

### Step 1: Open a terminal

Change directory to the folder containing the Python script and CSV files.

### Step 2: Create a virtual environment (recommended)

**Windows (Command Prompt):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn plotly
```

You can also save these packages in a `requirements.txt` file:

```text
streamlit
pandas
numpy
scikit-learn
plotly
```

Then install them with:

```bash
python -m pip install -r requirements.txt
```

## Run the dashboard

From the project directory, run:

```bash
streamlit run streamlit_app.py
```

If you are using the commented filename instead, run:

```bash
streamlit run streamlit_app_commented.py
```

Streamlit normally opens the dashboard in your web browser. If it does not, use the **Local URL** printed in the terminal (commonly `http://localhost:8501`).

To stop the application, focus the terminal and press **Ctrl+C**.

## Understanding the main calculations

### Expected AC power

The model uses these input features:

- `IRRADIATION`
- `AMBIENT_TEMPERATURE`
- `MODULE_TEMPERATURE`

The target is `AC_POWER`. A Random Forest regressor is trained on a subset of the records and used to estimate expected power.

### Power-loss percentage

When expected power is greater than zero, the application calculates:

\[
\text{Loss (\%)} =
\frac{\text{Expected Power} - \text{Actual AC Power}}
{\text{Expected Power}} \times 100
\]

If expected power is zero or below, the code assigns a loss value of zero to avoid division by zero. A negative loss means actual power exceeded the model's estimate; it does not necessarily indicate a problem.

### Anomaly detection

Isolation Forest receives `AC_POWER`, `DC_POWER`, `IRRADIATION`, and `MODULE_TEMPERATURE`. Its output is:

- `1` — the record is treated as normal by the model.
- `-1` — the record is flagged as anomalous.

The `contamination=0.03` setting tells the model to expect approximately 3% anomalous records. This is a configuration assumption, not proof that exactly 3% of real measurements are faulty.

### Health score and failure-risk indicator

The application starts the risk score from positive power loss, adds 50 points when a record is anomalous, and caps the score at 100. Health is calculated as:

\[
\text{Health Score} = 100 - \text{Risk Score}
\]

The failure-risk indicator starts at `100 - HEALTH_SCORE`. It then adds 10 points when module temperature is above 55°C and 15 points when the record is anomalous, with an upper limit of 100.

These rules are manually chosen thresholds. The resulting number is a **risk indicator**, not a calibrated probability derived from confirmed failure history.

### Risk categories and suggested actions

The code assigns failure levels using these thresholds:

- **Low:** below 30
- **Medium:** 30 to below 60
- **High:** 60 or above

Suggested actions use a separate threshold scheme:

- **80 or above:** Immediate Inspection
- **60 to below 80:** Inspect Within 24 Hours
- **40 to below 60:** Schedule Maintenance
- **Below 40:** Monitor

These labels are intended to help organize review. Confirm the thresholds with qualified solar-plant operators before using them in a real maintenance workflow.

### Rule-based diagnosis

The application checks for patterns that may suggest:

- **Possible overheating:** power loss above 20% and module temperature above 55°C.
- **Possible dust or shading:** power loss above 20% and irradiation above 0.5.
- **Possible inverter fault:** DC power is more than 1.2 times AC power.
- **Normal:** none of the above rules match.

These are clues for further inspection, not definitive fault diagnoses.

## Troubleshooting

### `FileNotFoundError`

Make sure both CSV files are in the same working directory from which you run Streamlit. Check spelling and capitalization:

- `Plant_2_Generation_Data.csv`
- `Plant_2_Weather_Sensor_Data.csv`

### `KeyError` for a column name

Open the CSV files and verify their headers. The application expects the column names listed in [Project files and data requirements](#project-files-and-data-requirements). Also check the merged dataframe for `SOURCE_KEY_x`; pandas only creates suffixes for columns that overlap between the two input tables.

### No data appears after merging

The inner join requires matching `DATE_TIME` and `PLANT_ID` values in both files. Confirm that timestamps use compatible formats and that plant identifiers match. An inner join discards rows that have no match.

### `ValueError` when sampling rows for AI Predictions

The current code samples 2,000 rows for the scatter plot. If fewer than 2,000 cleaned records are available, sampling without replacement will fail. Use a sample size no greater than the number of available records, for example `min(2000, len(clean))`.

### Dashboard startup or package errors

Activate the intended virtual environment and install the dependencies listed above. If an error remains, read the full traceback in the terminal; the final lines usually identify the failing operation.

## Limitations and suggested improvements

Before relying on the dashboard for operational decisions, consider these improvements:

1. **Evaluate the model.** The code creates a test split but does not currently calculate test-set metrics. Report metrics such as MAE and RMSE on `X_test` and `y_test`.
2. **Avoid evaluating on training data.** The current expected-power predictions are generated for all cleaned rows, including rows used to train the model. Use held-out or time-based validation to understand how well the model generalizes.
3. **Use time-aware validation.** Solar readings are time-dependent. A chronological split may provide a more realistic assessment than a random split.
4. **Validate risk scores.** Compare thresholds and alerts against confirmed maintenance records and actual failures; calibrate the risk model if a meaningful failure probability is required.
5. **Handle small datasets and missing columns.** Check that required columns exist, that the merged dataset is not empty, and that enough rows remain for training and visualization.
6. **Avoid modifying a dataframe view unintentionally.** Consider making an explicit copy after filtering, for example `clean = data.dropna(subset=features + [target]).copy()`.
7. **Consider retraining strategy.** For a production application, define when the model is retrained and monitor model drift as equipment or weather conditions change.
8. **Separate inference from dashboard rendering.** Moving data preparation and model training into well-defined functions can make the project easier to test and maintain.

## Disclaimer

This project is a data-analysis and dashboard prototype. Model estimates, anomaly flags, health scores, diagnosis labels, and maintenance suggestions should be reviewed by qualified personnel and validated against reliable plant data before operational use.
