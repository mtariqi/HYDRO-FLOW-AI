<p align="center">
  <img src="docs/assets/HYDRO-FLOW-AI-banner.png" alt="HYDRO-FLOW-AI banner" width="100%">
</p>

<h1 align="center">HYDRO-FLOW-AI</h1>

<p align="center">
  <strong>Extreme-aware AI for streamflow prediction, National Water Model bias correction, uncertainty quantification, and flood-event detection.</strong>
</p>

<p align="center">
  <a href="https://github.com/mtariqi/HYDRO-FLOW-AI">
    <img src="https://img.shields.io/badge/GitHub-mtariqi%2FHYDRO--FLOW--AI-181717?logo=github" alt="GitHub repository">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/status-active%20research-2ea44f" alt="Active research">
  <img src="https://img.shields.io/badge/domain-hydrology-0077B6" alt="Hydrology">
  <img src="https://img.shields.io/badge/focus-extreme%20streamflow-D1495B" alt="Extreme streamflow">
  <img src="https://img.shields.io/badge/model-XGBoost%20baseline-EB5E28" alt="XGBoost baseline">
</p>

<p align="center">
  <a href="https://github.com/mtariqi/HYDRO-FLOW-AI/stargazers">
    <img src="https://img.shields.io/github/stars/mtariqi/HYDRO-FLOW-AI?style=flat&logo=github" alt="GitHub stars">
  </a>
  <a href="https://github.com/mtariqi/HYDRO-FLOW-AI/network/members">
    <img src="https://img.shields.io/github/forks/mtariqi/HYDRO-FLOW-AI?style=flat&logo=github" alt="GitHub forks">
  </a>
  <a href="https://github.com/mtariqi/HYDRO-FLOW-AI/issues">
    <img src="https://img.shields.io/github/issues/mtariqi/HYDRO-FLOW-AI?style=flat&logo=github" alt="GitHub issues">
  </a>
  <img src="https://img.shields.io/github/last-commit/mtariqi/HYDRO-FLOW-AI?style=flat&logo=git" alt="Last commit">
  <img src="https://img.shields.io/badge/reproducibility-priority-6f42c1" alt="Reproducibility priority">
  <img src="https://img.shields.io/badge/data-USGS%20%2B%20Climate%20%2B%20NWM-0A9396" alt="Data sources">
</p>

# Why HYDRO-FLOW-AI?

National Water Model (NWM) forecasts are useful but can be systematically biased at individual gauges, and that bias can be especially consequential during high-flow and flood events. HYDRO-FLOW-AI treats bias correction as more than a generic regression task: the framework is explicitly designed to evaluate and improve performance in the tails of the streamflow distribution, not only average error under mostly normal conditions.

The current research baseline reconstructs a dated, site-aware training dataset and prepares a leakage-safe path toward temporal feature engineering, residual correction, extreme-event metrics, uncertainty quantification, and later spatiotemporal graph learning.

Study sites

Two USGS gauges are used in the current reconstruction workflow.

NWIS Site ID

USGS station

Latitude

Longitude

10133800

East Canyon Creek near Jeremy Ranch, UT

40.75966979

-111.5640912

10133600

McLeod Creek near Park City, UT

40.68803889

-111.5337194

Observed record: 1979–2023, daily. The current workflow combines observed USGS streamflow, temperature, precipitation, basin/site characteristics, and NWM flow. The dated reconstruction has been validated at 5,740 / 5,740 rows, with zero missing dates and zero duplicate (site, datetime) pairs.

End-to-end research pipeline

<img width="864" height="571" alt="image" src="https://github.com/user-attachments/assets/451a2c61-0ff2-41cb-bc71-cc98da9fb255" />




# PIPELINE LOGIC

**Extreme-aware AI framework for streamflow prediction and NWM bias correction**, built on hydrometeorological data, temporal feature engineering, machine learning, and uncertainty-aware flood-event detection.

## Motivation

National Water Model (NWM) forecasts are useful but systematically biased at individual gauges, and that bias is often worst exactly when it matters most — during high-flow and flood events. This project treats bias correction as more than a generic regression problem: the goal is a model that is explicitly evaluated on its ability to represent extremes, not just to minimize average error across mostly-normal conditions.

## Study sites

Two USGS gauges, matched to NWM grid cells by coordinates:

| NWIS Site ID | Latitude | Longitude |
|---|---:|---:|
| 10133800 | 40.75966979 | -111.5640912 |
| 10133600 | 40.68803889 | -111.5337194 |

Observed record: 2012–2019 (daily), climate forcing (temperature, precipitation) and observed USGS streamflow, joined against NWM output.

## Pipeline

```mermaid
flowchart TD
    subgraph UP["Upstream data — external NWM-ML repo, not tracked here"]
        A1["NWM_ML_Training_DF.csv"]
        A2["Climate.csv"]
        A3["flow.pkl"]
    end

    subgraph S1["Stage 1 · Dataset Reconstruction — done"]
        B1["assign_site_from_coordinates()<br/>Lat/Long to NWIS_site_id"]
        B2["build_source_table()<br/>Climate + observed flow joined on datetime, site"]
        B3["reconstruct_dates()<br/>match on site, day-of-year, tolerance on T/P/Q"]
        B4{"one-to-one?<br/>5740 matched, 0 zero-match, 0 multi-match"}
        B5[["nwm_ml_dated.csv"]]
        B6[["reconstruction_diagnostics.csv"]]
        B7(["ReconstructionError: refuses to write output"])
    end

    A1 --> B1
    B1 --> B3
    A2 --> B2
    A3 --> B2
    B2 --> B3
    B3 --> B4
    B4 -->|pass| B5
    B4 -->|fail| B6
    B4 -->|fail| B7

    subgraph S2["Stage 2 · Temporal Feature Engineering — next"]
        C1["NWM discharge lags: Qt, Qt-1, Qt-3, Qt-7"]
        C2["Precipitation sums: Pt, P3d, P7d, P14d"]
        C3["Temperature lags: Tt, Tt-1, Tt-3, Tt-7"]
        C4["Cyclic seasonality: sin/cos(DOY)"]
        C5["Basin / site characteristics"]
        C6[["feature_matrix.csv"]]
    end

    B5 --> C1
    B5 --> C2
    B5 --> C3
    B5 --> C4
    B5 --> C5
    C1 --> C6
    C2 --> C6
    C3 --> C6
    C4 --> C6
    C5 --> C6

    subgraph S3["Stage 3 · Leakage-safe split"]
        D1["2012-2017: TRAIN"]
        D2["2018: VALIDATION"]
        D3["2019: TEST, untouched"]
    end

    C6 --> D1
    C6 --> D2
    C6 --> D3

    subgraph S4["Stage 4 · Modeling"]
        E1["NWM baseline, no correction"]
        E2["XGBoost residual model<br/>residual = Q_obs - Q_NWM"]
        E3["corrected = Q_NWM + predicted_residual"]
    end

    D1 --> E2
    D2 --> E2
    E2 --> E3
    E1 -. compared against .-> E3

    subgraph S5["Stage 5 · Extreme-aware evaluation"]
        F1["RMSE / MAE / Bias"]
        F2["NSE / KGE"]
        F3["Q95 / Q99 error"]
        F4["Peak-magnitude error"]
        F5["Extreme-event detection"]
    end

    D3 --> E3
    E3 --> F1
    E3 --> F2
    E3 --> F3
    E3 --> F4
    E3 --> F5

    subgraph S6["Stage 6 · Advanced & uncertainty — planned"]
        G1["Quantile regression,<br/>calibrated prediction intervals"]
        G2["Transformer / LSTM"]
        G3["River-network GNN"]
        G4["Probabilistic extreme-event head"]
    end

    F1 --> G1
    F2 --> G1
    F3 --> G1
    F4 --> G1
    F5 --> G1
    G1 --> G2
    G2 --> G3
    G3 --> G4

    classDef done fill:#1b4332,stroke:#2d6a4f,color:#ffffff
    classDef next fill:#023047,stroke:#219ebc,color:#ffffff
    classDef planned fill:#3c3744,stroke:#7a7788,color:#ffffff
    classDef fail fill:#7f1d1d,stroke:#b91c1c,color:#ffffff

    class B1,B2,B3,B4,B5,B6 done
    class B7 fail
    class C1,C2,C3,C4,C5,C6,D1,D2,D3,E1,E2,E3,F1,F2,F3,F4,F5 next
    class G1,G2,G3,G4 planned
```

**Legend:** green = complete and validated · blue = next / actively planned near-term · gray = longer-horizon roadmap · red = the validation-failure path (`ReconstructionError`, output withheld).

### 1. Dataset reconstruction (complete)

`src/hydro_flow_ai/reconstruct_dataset.py` recovers the `datetime` and `NWIS_site_id` fields that were dropped from the original ML training export (`NWM_ML_Training_DF.csv`), by:

1. Mapping each row back to its gauge using retained Lat/Long coordinates.
2. Rebuilding a dated source-of-truth table from `Climate.csv` + observed USGS flow (`flow.pkl`).
3. Restricting candidate dates to the same (site, day-of-year), then matching on temperature, precipitation, and flow using an absolute floating-point tolerance (`--match-atol`, default `1e-5`) rather than exact equality, to absorb floating-point noise introduced between the original export and the current source files.
4. Requiring every training row to match **exactly one** candidate date — the script refuses to write output otherwise, and writes a diagnostics CSV explaining any row that failed to match uniquely.

Status: verified end-to-end on both gauges — all 5,740 training rows (2,827 + 2,913) reconstruct to a unique date with zero ambiguous or missing matches.

```fish
python src/hydro_flow_ai/reconstruct_dataset.py
```

### 2. Temporal feature engineering (next)

Planned split — chosen specifically to avoid temporal leakage rather than a random split, since random splitting on time series data inflates apparent performance on extreme events:

- **2012–2017** → training
- **2018** → validation
- **2019** → held out, untouched test set

Planned initial feature set, restricted to variables actually available at prediction time (no future leakage):

- NWM discharge: `Q_t`, `Q_{t-1}`, `Q_{t-3}`, `Q_{t-7}`
- Precipitation: `P_t`, and rolling sums `P_3d`, `P_7d`, `P_14d`
- Temperature: `T_t`, `T_{t-1}`, `T_{t-3}`, `T_{t-7}`
- Basin/site characteristics
- Cyclic seasonality encoding (e.g. day-of-year as sin/cos)

### 3. Modeling and evaluation (planned)

Baseline model: XGBoost residual correction, where the model learns

```text
residual = observed_streamflow - NWM_streamflow
```

and the corrected forecast is

```text
corrected_streamflow = NWM_streamflow + predicted_residual
```

Evaluation deliberately goes beyond average-case accuracy:

- Standard: RMSE, MAE, Bias, NSE, KGE
- Extreme-focused: Q95/Q99 RMSE, peak-magnitude error, extreme-event detection performance

## Project structure

```text
HYDRO-FLOW-AI/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── derived/
├── src/
│   └── hydro_flow_ai/
│       ├── __init__.py
│       └── reconstruct_dataset.py
├── tests/
├── configs/
├── notebooks/
├── models/
├── results/
│   ├── figures/
│   ├── tables/
│   └── metrics/
├── docs/
│   └── assets/
│       └── HYDRO-FLOW-AI-banner.png
├── README.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Setup
### 4. Quick Start & Setup
1. Environment Setup
Clone the repository and set up a virtual environment:
```
git clone [https://github.com/your-username/HYDRO-FLOW-AI.git](https://github.com/your-username/HYDRO-FLOW-AI.git)
cd HYDRO-FLOW-AI

# Create virtual environment
python -m venv .venv

# Activate environment (bash/zsh)
source .venv/bin/activate

# Activate environment (fish)
# source .venv/bin/activate.fish

# Install dependencies
pip install -r requirements.txt
```fish
2. Run Reconstruction Pipeline
Execute Stage 1 data reconstruction:
python src/hydro_flow_ai/reconstruct_dataset.py

## Upstream reference

This project extends ideas and data-processing workflows from the Alabama Water Institute NWM-ML project. HYDRO-FLOW-AI is an independent research repository focused on extreme-flow bias correction, leakage-safe temporal modeling, uncertainty quantification, and future spatiotemporal graph learning.

## Status

- [x] Dataset reconstruction — datetime/site recovery, validated one-to-one (5,740/5,740 rows)
- [ ] Temporal feature engineering
- [ ] Leakage-safe train/validation/test split
- [ ] Baseline XGBoost residual-correction model
- [ ] Extreme-event evaluation suite (Q95/Q99, peak error, detection metrics)
- [ ] Uncertainty quantification using quantile regression and calibrated prediction intervals
- [ ] Transformer/LSTM temporal model
- [ ] River-network graph neural network
- [ ] Probabilistic extreme-event detection head
- [ ] Model comparison and ablation study
