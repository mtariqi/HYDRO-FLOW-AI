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

Why HYDRO-FLOW-AI?

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

Observed record: 2012–2019, daily. The current workflow combines observed USGS streamflow, temperature, precipitation, basin/site characteristics, and NWM flow. The dated reconstruction has been validated at 5,740 / 5,740 rows, with zero missing dates and zero duplicate (site, datetime) pairs.

End-to-end research pipeline

flowchart LR
    classDef source fill:#eaf4ff,stroke:#3b82f6,color:#0f172a,stroke-width:1px;
    classDef reconstruct fill:#ecfdf5,stroke:#10b981,color:#0f172a,stroke-width:1px;
    classDef feature fill:#fff7ed,stroke:#f59e0b,color:#0f172a,stroke-width:1px;
    classDef split fill:#f5f3ff,stroke:#8b5cf6,color:#0f172a,stroke-width:1px;
    classDef model fill:#fff1f2,stroke:#f43f5e,color:#0f172a,stroke-width:1px;
    classDef eval fill:#f0fdfa,stroke:#14b8a6,color:#0f172a,stroke-width:1px;
    classDef future fill:#eef2ff,stroke:#6366f1,color:#0f172a,stroke-width:1px;

    subgraph A["1 · Data sources"]
        A1["USGS observed<br/>streamflow"]:::source
        A2["Climate forcing<br/>temperature + precipitation"]:::source
        A3["NWM flow"]:::source
        A4["Basin / site<br/>attributes"]:::source
    end

    subgraph B["2 · Dataset reconstruction"]
        B1["Recover NWIS site ID<br/>from retained coordinates"]:::reconstruct
        B2["Rebuild dated source table<br/>Climate + USGS flow"]:::reconstruct
        B3["Candidate filter<br/>site + day-of-year"]:::reconstruct
        B4["Tolerance match<br/>T + P + Qobs"]:::reconstruct
        B5["1:1 validation<br/>5,740 unique rows"]:::reconstruct
    end

    subgraph C["3 · Temporal feature engineering"]
        C1["NWM lags<br/>Qt, Qt-1, Qt-3, Qt-7"]:::feature
        C2["Precipitation<br/>Pt, P3d, P7d, P14d"]:::feature
        C3["Temperature lags<br/>Tt, Tt-1, Tt-3, Tt-7"]:::feature
        C4["Seasonality<br/>sin/cos(DOY)"]:::feature
        C5["Static basin / site<br/>characteristics"]:::feature
    end

    subgraph D["4 · Leakage-safe temporal split"]
        D1["2012–2017<br/>TRAIN"]:::split
        D2["2018<br/>VALIDATION"]:::split
        D3["2019<br/>TEST · untouched"]:::split
    end

    subgraph E["5 · Modeling"]
        E1["Raw NWM<br/>reference baseline"]:::model
        E2["XGBoost<br/>residual correction"]:::model
        E3["residual = Qobs − QNWM"]:::model
        E4["Qcorrected = QNWM + residual_hat"]:::model
    end

    subgraph F["6 · Extreme-aware evaluation"]
        F1["Standard<br/>RMSE · MAE · Bias"]:::eval
        F2["Hydrology<br/>NSE · KGE"]:::eval
        F3["Extreme tails<br/>Q95 / Q99 RMSE"]:::eval
        F4["Event skill<br/>POD · FAR · CSI"]:::eval
        F5["Peak magnitude<br/>+ timing error"]:::eval
    end

    subgraph G["7 · Uncertainty + advanced AI"]
        G1["Quantile regression<br/>prediction intervals"]:::future
        G2["Conformal calibration"]:::future
        G3["LSTM / Transformer"]:::future
        G4["River-network GNN"]:::future
        G5["Probabilistic<br/>extreme-event head"]:::future
    end

    A1 --> B1
    A2 --> B2
    A3 --> C1
    A4 --> C5

    B1 --> B2 --> B3 --> B4 --> B5
    B5 --> C1
    B5 --> C2
    B5 --> C3
    B5 --> C4
    B5 --> C5

    C1 --> D1
    C2 --> D1
    C3 --> D1
    C4 --> D1
    C5 --> D1

    D1 --> E1
    D1 --> E2
    D2 --> E2
    E1 --> F1
    E2 --> E3 --> E4 --> F1

    F1 --> F2 --> F3 --> F4 --> F5
    D3 --> F1

    F5 --> G1 --> G2
    G2 --> G3
    G2 --> G4
    G3 --> G5
    G4 --> G5

Pipeline logic

Stage

Purpose

Inputs

Output

1. Data sources

Assemble observed and modeled hydrology

USGS flow, climate, NWM, static basin data

Raw research inputs

2. Reconstruction

Restore temporal/site identity lost in the original export

Lat/Long, DOY, climate, observed flow

Dated, site-aware dataset

3. Feature engineering

Represent antecedent hydrologic state without future leakage

NWM, precipitation, temperature, seasonality

Model-ready temporal features

4. Temporal split

Prevent random-split leakage

Dated features

Train / validation / untouched test

5. Modeling

Learn systematic NWM residual error

NWM + features

Bias-corrected streamflow

6. Evaluation

Measure both average and tail performance

Observed + predicted flow

Standard + extreme-event metrics

7. Advanced AI

Add uncertainty, sequence learning, and network structure

Calibrated features + river graph

Probabilistic extreme-flow forecasts

Stage 1 — Dataset reconstruction ✅

src/hydro_flow_ai/reconstruct_dataset.py restores the datetime and NWIS_site_id fields that were dropped from the original ML training export.

The reconstruction:

Maps each row to an NWIS gauge from retained coordinates.

Rebuilds a dated source-of-truth table from Climate.csv and observed USGS flow in flow.pkl.

Restricts candidate dates to matching (site, day-of-year).

Matches temperature, precipitation, and observed flow using an absolute numeric tolerance (--match-atol, default 1e-5).

Requires exactly one recovered date per original training row.

Rejects ambiguous or zero-match rows and emits diagnostics rather than silently assigning a date.

Verifies the final (NWIS_site_id, datetime) mapping is one-to-one.

Validated result:

Original training rows : 5740
Unique matched rows    : 5740
Problem row IDs        : 0
Missing dates          : 0
Duplicate site/date    : 0

Run:

python src/hydro_flow_ai/reconstruct_dataset.py

Stage 2 — Temporal feature engineering 🚧

The initial feature set is deliberately restricted to information available at prediction time.

NWM discharge

[
Q_t,\quad Q_{t-1},\quad Q_{t-3},\quad Q_{t-7}
]

Antecedent precipitation

[
P_t,\quad P_{3d},\quad P_{7d},\quad P_{14d}
]

where the rolling precipitation totals summarize antecedent watershed forcing.

Temperature

[
T_t,\quad T_{t-1},\quad T_{t-3},\quad T_{t-7}
]

Seasonality

[
\sin(2\pi,DOY/365.25),\qquad
\cos(2\pi,DOY/365.25)
]

Static context

drainage area

mean basin elevation

forest / developed / impervious cover

herbaceous cover

steep-slope fraction

mean annual precipitation

site identity

Leakage-safe experiment design

flowchart LR
    A["2012–2017<br/><b>TRAIN</b>"]
    B["2018<br/><b>VALIDATION</b>"]
    C["2019<br/><b>HELD-OUT TEST</b>"]

    A -->|"fit model + learn preprocessing"| B
    B -->|"select hyperparameters"| C
    C -->|"final evaluation only"| D["Report final<br/>standard + extreme metrics"]

    style A fill:#dcfce7,stroke:#16a34a
    style B fill:#fef3c7,stroke:#d97706
    style C fill:#fee2e2,stroke:#dc2626
    style D fill:#dbeafe,stroke:#2563eb

A random split is intentionally avoided because adjacent time-series observations are autocorrelated and random splitting can make extreme-event performance appear unrealistically strong.

Stage 3 — Residual correction baseline

The initial ML baseline is XGBoost residual correction.

[
e_t = Q^{obs}_t - Q^{NWM}_t
]

The model learns:

[
\hat e_t = f_\theta(X_t)
]

and produces:

Q^{NWM}_t + \hat e_t
]

This preserves the NWM prediction as the physical/model baseline while asking ML to learn systematic error conditioned on hydrometeorological state.

Extreme-aware evaluation

Average metrics alone can hide failures at flood peaks. HYDRO-FLOW-AI therefore separates ordinary performance from tail performance.

Standard hydrologic metrics

Metric

Role

RMSE

overall squared-error magnitude

MAE

robust average absolute error

Bias

systematic over/underprediction

NSE

hydrologic efficiency

KGE

correlation, variability, and bias balance

Extreme-flow metrics

Metric

Role

Q95 RMSE

performance above the 95th percentile

Q99 RMSE

performance above the 99th percentile

Peak magnitude error

under/overprediction of peak discharge

Peak timing error

temporal displacement of peaks

POD

probability of detecting an extreme event

FAR

false-alarm ratio

CSI

critical success index

Precision / Recall

class imbalance-aware event detection

Planned uncertainty quantification

The next uncertainty layer will investigate:

quantile-regression objectives

calibrated prediction intervals

conformal prediction

interval coverage and sharpness

reliability / calibration diagnostics

probabilistic threshold exceedance

The eventual probabilistic task is:

[
P(Q_{t+h} > Q_{\text{critical}})
]

rather than returning only a single deterministic discharge estimate.

Advanced model roadmap

flowchart TD
    A["Extreme-aware XGBoost<br/>residual baseline"]
    B["Temporal sequence model<br/>LSTM / GRU"]
    C["Transformer / N-HiTS<br/>multi-frequency temporal encoder"]
    D["Directed river-network GNN<br/>upstream → downstream"]
    E["Hybrid spatiotemporal model"]
    F["Probabilistic extreme-event head"]
    G["Uncertainty calibration"]
    H["Ablation + model comparison"]

    A --> B
    A --> C
    B --> E
    C --> E
    D --> E
    E --> F --> G --> H

    style A fill:#fff7ed,stroke:#ea580c
    style B fill:#eff6ff,stroke:#2563eb
    style C fill:#eff6ff,stroke:#2563eb
    style D fill:#f5f3ff,stroke:#7c3aed
    style E fill:#ecfdf5,stroke:#059669
    style F fill:#fff1f2,stroke:#e11d48
    style G fill:#f0fdfa,stroke:#0f766e
    style H fill:#f8fafc,stroke:#475569

Repository structure

HYDRO-FLOW-AI/
├── configs/
├── data/
│   ├── raw/                  # gitignored source inputs
│   ├── interim/              # gitignored intermediate outputs
│   ├── processed/            # gitignored cleaned data
│   └── derived/              # gitignored reconstructed/features
├── docs/
│   └── assets/
│       └── HYDRO-FLOW-AI-banner.png
├── models/                   # gitignored trained artifacts
├── notebooks/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── tables/
├── src/
│   └── hydro_flow_ai/
│       ├── __init__.py
│       └── reconstruct_dataset.py
├── tests/
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt

Quick start

1. Clone

git clone https://github.com/mtariqi/HYDRO-FLOW-AI.git
cd HYDRO-FLOW-AI

2. Create an environment

For fish shell:

python3 -m venv .venv
source .venv/bin/activate.fish

For Bash:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Reconstruct the dated dataset

The reconstruction script requires the corresponding source data files locally.

python src/hydro_flow_ai/reconstruct_dataset.py

Expected validation:

Original training rows : 5740
Unique matched rows    : 5740
Problem row IDs        : 0

Reproducibility principles

No random temporal split for the primary experiment.

No future information in predictor construction.

Untouched final test period for final reporting.

Extreme-flow metrics reported separately from overall metrics.

Ambiguous reconstruction fails loudly instead of silently guessing.

Source and derived data are not committed when they are large, restricted, or reproducibly generated.

NWM baseline remains visible in every model comparison.

Ablation studies will quantify the value of each architectural addition.

Upstream reference

This repository extends ideas and data-processing workflows from the Alabama Water Institute NWM-ML project.

HYDRO-FLOW-AI is maintained as an independent research repository focused on:

extreme-flow bias correction

leakage-safe temporal modeling

uncertainty quantification

probabilistic event detection

future spatiotemporal river-network learning

The upstream project should be cited and acknowledged where its code, workflow concepts, or data-processing approach materially contribute to derived work.

Project status

Component

Status

Dataset reconstruction

✅ Complete — 5,740 / 5,740 uniquely dated

Independent reconstruction validation

✅ Complete

Temporal feature engineering

🚧 Next

Leakage-safe train/validation/test split

⏳ Planned

XGBoost residual-correction baseline

⏳ Planned

Extreme-event evaluation suite

⏳ Planned

Uncertainty quantification

⏳ Planned

LSTM / Transformer temporal model

⏳ Planned

River-network GNN

⏳ Planned

Probabilistic extreme-event head

⏳ Planned

Model ablation study

⏳ Planned

Technology stack

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-data%20engineering-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-numerical%20computing-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-baseline-EB5E28" alt="XGBoost">
  <img src="https://img.shields.io/badge/hydroeval-hydrologic%20metrics-0077B6" alt="hydroeval">
  <img src="https://img.shields.io/badge/Matplotlib-visualization-11557C" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Ruff-linting-D7FF64?logo=ruff&logoColor=black" alt="Ruff">
</p>

Citation

A formal project citation will be added with the first tagged research release. Until then, please cite the repository URL and the upstream NWM-ML work where appropriate.

Contributing

Issues, reproducibility checks, feature requests, hydrology-domain feedback, and model-comparison contributions are welcome.

For substantial changes, open an issue first describing:

the hydrologic or ML problem,

the proposed change,

expected validation,

whether the change affects leakage, uncertainty, or extreme-event evaluation.

<p align="center">
  <strong>HYDRO-FLOW-AI</strong><br/>
  Better bias correction is useful. Better extreme-event prediction is the goal.
</p>
