<p align="center">
  <img src="docs/assets/HYDRO-FLOW-AI-banner.png" alt="HYDRO-FLOW-AI banner" width="100%">
</p>

<h1 align="center">HYDRO-FLOW-AI</h1>

<p align="center">
  <strong>
    Extreme-aware AI for streamflow prediction, National Water Model bias correction,
    uncertainty quantification, and flood-event detection.
  </strong>
</p>

<p align="center">
  <a href="https://github.com/mtariqi/HYDRO-FLOW-AI">
    <img src="https://img.shields.io/badge/GitHub-mtariqi%2FHYDRO--FLOW--AI-181717?logo=github" alt="GitHub repository">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/status-active%20research-2ea44f" alt="Active research">
  <img src="https://img.shields.io/badge/domain-hydrology-0077B6" alt="Hydrology">
  <img src="https://img.shields.io/badge/focus-extreme%20streamflow-D1495B" alt="Extreme streamflow">
  <img src="https://img.shields.io/badge/baseline-XGBoost-EB5E28" alt="XGBoost baseline">
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

---

## Why HYDRO-FLOW-AI?

National Water Model (NWM) forecasts provide an important hydrologic baseline, but local forecast errors can become substantial at individual gauges. Those errors are particularly important during high-flow and flood-related conditions, when underprediction or mistimed peaks can reduce the usefulness of operational forecasts.

HYDRO-FLOW-AI treats NWM bias correction as more than a conventional regression problem. The framework is designed to:

* preserve the NWM prediction as a physical/model baseline;
* learn systematic residual error from hydrometeorological context;
* represent antecedent rainfall, temperature, seasonality, and watershed characteristics;
* avoid temporal leakage during training and validation;
* evaluate high-flow behavior separately from average-flow performance;
* quantify predictive uncertainty;
* support future temporal deep learning and river-network graph learning.

The central research question is:

> **Can extreme-aware, temporally informed AI improve NWM streamflow predictions without sacrificing reproducibility, hydrologic interpretability, or out-of-sample validity?**

---
<img width="1209" height="965" alt="image" src="https://github.com/user-attachments/assets/f9cc14f1-50a6-4d68-99c5-8be7456d8f0b" />

## Study sites

The current validated reconstruction uses two USGS gauges in Utah.

| NWIS Site ID | Station                                 |    Latitude |    Longitude |
| ------------ | --------------------------------------- | ----------: | -----------: |
| `10133800`   | East Canyon Creek near Jeremy Ranch, UT | 40.75966979 | -111.5640912 |
| `10133600`   | McLeod Creek near Park City, UT         | 40.68803889 | -111.5337194 |

### Current modeling record

The reconstructed machine-learning dataset spans **2012–2019 at daily resolution**.

The current workflow combines:

* observed USGS streamflow;
* NWM streamflow;
* temperature;
* precipitation;
* drainage area;
* basin elevation;
* forest/developed/impervious/herbaceous cover;
* steep-slope fraction;
* mean annual precipitation;
* seasonal timing information.

The validated reconstruction contains:

```text
Total rows              : 5,740
NWIS 10133600            : 2,827
NWIS 10133800            : 2,913
Missing dates            : 0
Duplicate site/date rows : 0
Ambiguous date matches   : 0
```

---

# Research pipeline
```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontSize": "24px",
    "fontFamily": "Arial, Helvetica, sans-serif",
    "primaryTextColor": "#111827",
    "lineColor": "#475569"
  },
  "flowchart": {
    "htmlLabels": true,
    "nodeSpacing": 70,
    "rankSpacing": 80,
    "curve": "basis"
  }
}}%%

flowchart TD

    subgraph INPUTS["1 · Hydrologic and Environmental Inputs"]
        direction LR
        A1["USGS observed<br/>streamflow"]
        A2["Climate forcing<br/>temperature + precipitation"]
        A3["NWM streamflow"]
        A4["Basin and site<br/>attributes"]
    end

    subgraph RECON["2 · Dataset Reconstruction · COMPLETE"]
        direction TB

        B1["Map retained coordinates<br/>to NWIS site ID"]

        B2["Build dated source table<br/>Climate + USGS flow"]

        B3["Restrict candidate dates<br/>site + day-of-year"]

        B4["Tolerance-based match<br/>temperature + precipitation + flow"]

        B5{"Exactly one date<br/>per training row?"}

        B6["Validated dated dataset<br/><b>5,740 / 5,740 rows</b>"]

        B7["Diagnostics + ReconstructionError<br/>output withheld"]

        B1 --> B3
        B2 --> B3
        B3 --> B4
        B4 --> B5
        B5 -->|"PASS"| B6
        B5 -->|"FAIL"| B7
    end

    subgraph FEATURES["3 · Temporal Feature Engineering · NEXT"]
        direction LR

        C1["NWM flow<br/>Qt, Qt−1, Qt−3, Qt−7"]

        C2["Precipitation<br/>Pt, P3d, P7d, P14d"]

        C3["Temperature<br/>Tt, Tt−1, Tt−3, Tt−7"]

        C4["Cyclic seasonality<br/>sin(DOY), cos(DOY)"]

        C5["Static watershed<br/>context"]

        C6["Model-ready<br/>feature matrix"]
    end

    subgraph SPLIT["4 · Leakage-Safe Temporal Split"]
        direction LR

        D1["2012–2017<br/><b>TRAIN</b>"]
        D2["2018<br/><b>VALIDATION</b>"]
        D3["2019<br/><b>HELD-OUT TEST</b>"]
    end

    subgraph MODEL["5 · Modeling"]
        direction LR

        E1["Raw NWM<br/>baseline"]

        E2["Residual target<br/>Qobs − QNWM"]

        E3["XGBoost<br/>residual model"]

        E4["Corrected forecast<br/>QNWM + residual_hat"]
    end

    subgraph EVAL["6 · Extreme-Aware Evaluation"]
        direction LR

        F1["RMSE · MAE · Bias"]
        F2["NSE · KGE"]
        F3["Q95 / Q99 RMSE"]
        F4["Peak magnitude<br/>error"]
        F5["Peak timing<br/>error"]
        F6["POD · FAR · CSI"]
    end

    subgraph UQ["7 · Uncertainty Quantification"]
        direction LR

        G1["Quantile<br/>regression"]
        G2["Prediction<br/>intervals"]
        G3["Conformal<br/>calibration"]
        G4["Threshold-exceedance<br/>probability"]
    end

    subgraph ADV["8 · Advanced AI Roadmap"]
        direction LR

        H1["LSTM / GRU"]
        H2["Transformer / N-HiTS"]
        H3["Directed river-network<br/>GNN"]
        H4["Hybrid spatiotemporal<br/>model"]
        H5["Probabilistic<br/>extreme-event head"]
        H6["Ablation +<br/>model comparison"]
    end

    A1 --> B2
    A2 --> B2
    A3 --> C1
    A4 --> C5

    B6 --> C1
    B6 --> C2
    B6 --> C3
    B6 --> C4
    B6 --> C5

    C1 --> C6
    C2 --> C6
    C3 --> C6
    C4 --> C6
    C5 --> C6

    C6 --> D1
    C6 --> D2
    C6 --> D3

    D1 --> E3
    D2 --> E3
    E2 --> E3
    E3 --> E4

    E1 -. baseline comparison .-> E4
    D3 --> E4

    E4 --> F1
    E4 --> F2
    E4 --> F3
    E4 --> F4
    E4 --> F5
    E4 --> F6

    F3 --> G1
    F4 --> G1
    F6 --> G4

    G1 --> G2
    G2 --> G3

    G3 --> H1
    G3 --> H2
    H1 --> H4
    H2 --> H4
    H3 --> H4
    H4 --> H5
    H5 --> H6

    classDef input fill:#ede9fe,stroke:#7c3aed,color:#111827,stroke-width:2px,font-size:24px;
    classDef complete fill:#14532d,stroke:#166534,color:#ffffff,stroke-width:2px,font-size:24px;
    classDef active fill:#075985,stroke:#0369a1,color:#ffffff,stroke-width:2px,font-size:24px;
    classDef planned fill:#374151,stroke:#6b7280,color:#ffffff,stroke-width:2px,font-size:24px;
    classDef failure fill:#991b1b,stroke:#b91c1c,color:#ffffff,stroke-width:2px,font-size:24px;

    class A1,A2,A3,A4 input;
    class B1,B2,B3,B4,B5,B6 complete;
    class B7 failure;
    class C1,C2,C3,C4,C5,C6,D1,D2,D3,E1,E2,E3,E4,F1,F2,F3,F4,F5,F6 active;
    class G1,G2,G3,G4,H1,H2,H3,H4,H5,H6 planned;
```

**Legend:**
🟢 complete and validated · 🔵 near-term implementation · ⚫ longer-term roadmap · 🔴 validation failure path

---

## Stage 1 — Dataset reconstruction ✅

The original ML export, `NWM_ML_Training_DF.csv`, retained hydrologic and environmental variables but removed `datetime` and `NWIS_site_id`.

`src/hydro_flow_ai/reconstruct_dataset.py` restores those fields reproducibly.

### Reconstruction logic

1. Assign each training row to an NWIS gauge using retained latitude/longitude.
2. Reconstruct a dated reference table from:

   * `Climate.csv`
   * observed USGS streamflow in `flow.pkl`
3. Restrict candidate dates to the same:

   * NWIS site;
   * day-of-year.
4. Match:

   * temperature;
   * precipitation;
   * observed streamflow.
5. Use an absolute numeric tolerance:

```text
--match-atol 1e-5
```

6. Require exactly **one** valid date for every original training row.
7. Reject zero-match or multi-match cases instead of silently guessing.
8. Validate uniqueness of `(NWIS_site_id, datetime)`.

### Validated result

```text
Original training rows : 5740
Unique matched rows    : 5740
Problem row IDs        : 0
Missing dates          : 0
Duplicate site/date    : 0
```

Run:

```fish
python src/hydro_flow_ai/reconstruct_dataset.py
```

---

## Stage 2 — Temporal feature engineering 🚧

The next stage converts the dated dataset into a leakage-safe temporal feature matrix.

### NWM flow features

```text
NWM_flow_t
NWM_flow_lag1
NWM_flow_lag3
NWM_flow_lag7
```

### Antecedent precipitation

```text
precip_1d
precip_3d
precip_7d
precip_14d
```

The multi-day precipitation features are intended to represent antecedent watershed forcing rather than only same-day rainfall.

### Temperature features

```text
temperature_t
temperature_lag1
temperature_lag3
temperature_lag7
```

### Seasonal encoding

Day-of-year is represented cyclically:

[
DOY_{\sin} =
\sin\left(\frac{2\pi,DOY}{365.25}\right)
]

[
DOY_{\cos} =
\cos\left(\frac{2\pi,DOY}{365.25}\right)
]

### Static watershed features

Current variables include:

* drainage area;
* mean basin elevation;
* percentage forest cover;
* percentage developed land;
* percentage impervious surface;
* percentage herbaceous cover;
* percentage steep slope;
* mean annual precipitation.

---

## Leakage-safe experiment design

A random split is intentionally avoided because adjacent daily observations are temporally autocorrelated.

```mermaid
flowchart LR

    A["2012–2017<br/><b>TRAIN</b>"]
    B["2018<br/><b>VALIDATION</b>"]
    C["2019<br/><b>HELD-OUT TEST</b>"]
    D["Final research metrics"]

    A -->|"fit preprocessing + model"| B
    B -->|"model selection / tuning"| C
    C -->|"final evaluation only"| D

    style A fill:#dcfce7,stroke:#16a34a
    style B fill:#fef3c7,stroke:#d97706
    style C fill:#fee2e2,stroke:#dc2626
    style D fill:#dbeafe,stroke:#2563eb
```

This design protects the final reported performance from temporal leakage and repeated access to the test period.

---

## Stage 3 — Residual-correction baseline

The initial machine-learning baseline is **XGBoost residual correction**.

The target is:

[
e_t =
Q^{obs}_t - Q^{NWM}_t
]

The model learns:

[
\hat e_t = f_{\theta}(X_t)
]

The corrected prediction becomes:

[
\hat Q^{corrected}_t =
Q^{NWM}_t + \hat e_t
]

This preserves the NWM prediction as the reference hydrologic estimate while allowing ML to learn systematic local error.

---

# Extreme-aware evaluation

Average performance alone can conceal severe errors during rare high-flow events.

HYDRO-FLOW-AI therefore separates conventional metrics from extreme-flow metrics.

## Standard metrics

| Metric | Purpose                                                |
| ------ | ------------------------------------------------------ |
| RMSE   | penalizes large prediction errors                      |
| MAE    | average absolute prediction error                      |
| Bias   | systematic over- or underprediction                    |
| NSE    | hydrologic predictive efficiency                       |
| KGE    | combined correlation, variability, and bias assessment |

## Extreme-focused metrics

| Metric               | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| Q95 RMSE             | error above the 95th-percentile flow threshold |
| Q99 RMSE             | error in the most extreme tail                 |
| Peak magnitude error | accuracy of maximum discharge                  |
| Peak timing error    | timing displacement of observed peaks          |
| POD                  | probability of detection                       |
| FAR                  | false-alarm ratio                              |
| CSI                  | critical success index                         |
| Precision / Recall   | class-imbalance-aware event detection          |

---

# Uncertainty quantification

Future deterministic predictions will be extended with uncertainty-aware outputs.

Planned methods include:

* quantile regression;
* calibrated prediction intervals;
* conformal prediction;
* interval coverage;
* interval sharpness;
* reliability analysis;
* probability of threshold exceedance.

The eventual extreme-event prediction target is:

[
P(Q_{t+h} > Q_{critical})
]

rather than only a single deterministic discharge value.

---

# Advanced AI roadmap

```mermaid
flowchart TD

    A["Extreme-aware XGBoost<br/>residual baseline"]
    B["LSTM / GRU<br/>temporal sequence model"]
    C["Transformer / N-HiTS<br/>multi-scale temporal model"]
    D["Directed river-network GNN<br/>upstream → downstream"]
    E["Hybrid spatiotemporal architecture"]
    F["Probabilistic extreme-event head"]
    G["Uncertainty calibration"]
    H["Ablation study"]
    I["Final model comparison"]

    A --> B
    A --> C

    B --> E
    C --> E
    D --> E

    E --> F
    F --> G
    G --> H
    H --> I

    style A fill:#fff7ed,stroke:#ea580c
    style B fill:#eff6ff,stroke:#2563eb
    style C fill:#eff6ff,stroke:#2563eb
    style D fill:#f5f3ff,stroke:#7c3aed
    style E fill:#ecfdf5,stroke:#059669
    style F fill:#fff1f2,stroke:#e11d48
    style G fill:#f0fdfa,stroke:#0f766e
    style H fill:#f8fafc,stroke:#475569
    style I fill:#f8fafc,stroke:#334155
```

---

# Repository structure

```text
Google Drive — collaborative/source data
        │
        ├── NWM data
        ├── USGS streamflow
        ├── climate data
        ├── site/basin data
        └── collaborator-provided datasets
                │
                ▼
Local HYDRO-FLOW-AI/data/
        │
        ├── raw/          ← downloaded source data
        ├── interim/      ← intermediate transformations
        ├── processed/    ← cleaned datasets
        └── derived/      ← ML-ready datasets
                │
                ▼
GitHub — code + documentation only
```
# Data Sources

HYDRO-FLOW-AI integrates hydrologic, meteorological, and watershed
information from multiple sources.

## Data provenance

| Source | Data | Role |
|---|---|---|
| USGS NWIS | Observed streamflow | Ground-truth discharge |
| National Water Model (NWM) | Simulated streamflow | Baseline/model input |
| Climate forcing | Temperature and precipitation | Meteorological predictors |
| Basin/site attributes | Watershed characteristics | Static predictors |
| Collaborative research data | Project-specific hydrologic datasets | Model development and validation |

## Collaborative data

Additional project datasets are maintained in a shared Google Drive
workspace used by the research collaborators.

These source datasets are intentionally excluded from the public GitHub
repository. The repository contains the processing, reconstruction,
feature-engineering, modeling, and evaluation code required for the
HYDRO-FLOW-AI workflow.

## Current study sites

- USGS NWIS 10133800
- USGS NWIS 10133600

Study period: 2012–2019.
---

# Quick start

## 1. Clone the repository

```bash
git clone https://github.com/mtariqi/HYDRO-FLOW-AI.git
cd HYDRO-FLOW-AI
```

## 2. Create a virtual environment

### Bash / Zsh

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Fish

```fish
python3 -m venv .venv
source .venv/bin/activate.fish
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Add source data locally

Large source and derived datasets are intentionally excluded from Git.

Place or link the required input files according to the local project configuration before running the reconstruction pipeline.

## 5. Run dataset reconstruction

```bash
python src/hydro_flow_ai/reconstruct_dataset.py
```

Expected validation:

```text
Original training rows : 5740
Unique matched rows    : 5740
Problem row IDs        : 0
```

---

# Reproducibility principles

HYDRO-FLOW-AI follows several methodological constraints:

* no random temporal split for the primary experiment;
* no future information in predictor construction;
* an untouched final test period;
* standard and extreme-flow metrics reported separately;
* failed reconstruction is explicit rather than silently corrected;
* derived datasets are reproducibly generated;
* large data and model artifacts remain outside source control;
* raw NWM performance is retained as a comparison baseline;
* feature and architecture additions are evaluated using ablation studies;
* uncertainty calibration is treated separately from point-prediction accuracy.

---

# Data management

Large hydrologic inputs and derived datasets are not stored directly in the GitHub repository.

The repository tracks:

* source code;
* configuration;
* documentation;
* reproducible processing logic;
* tests;
* lightweight research outputs.

Data directories are retained structurally with `.gitkeep` files while substantive datasets remain gitignored.

---

# Collaborator

**Mohammad Shahabul Alam, PhD**
Assistant Professor, University of West Florida

* GitHub: [shahab122](https://github.com/shahab122)
* LinkedIn: [Mohammad Shahabul Alam](https://www.linkedin.com/in/md-shahabul-alam/)

---

# Upstream reference

This project extends ideas and data-processing workflows from the Alabama Water Institute **NWM-ML** project.

HYDRO-FLOW-AI is maintained as an independent research repository focused on:

* extreme-flow NWM bias correction;
* leakage-safe temporal modeling;
* extreme-event evaluation;
* uncertainty quantification;
* probabilistic threshold prediction;
* temporal deep learning;
* future river-network graph learning.

Where upstream code, data-processing workflows, or methodological components are reused or adapted, the original project should be appropriately acknowledged and cited.

---

# Project status

| Component                                | Status     |
| ---------------------------------------- | ---------- |
| Repository architecture                  | ✅ Complete |
| Dataset reconstruction                   | ✅ Complete |
| 5,740-row one-to-one date validation     | ✅ Complete |
| Site/date duplicate validation           | ✅ Complete |
| Temporal feature engineering             | 🚧 Next    |
| Leakage-safe train/validation/test split | ⏳ Planned  |
| Raw NWM benchmark                        | ⏳ Planned  |
| XGBoost residual-correction baseline     | ⏳ Planned  |
| Standard hydrologic metrics              | ⏳ Planned  |
| Q95/Q99 extreme-event evaluation         | ⏳ Planned  |
| Peak magnitude / timing evaluation       | ⏳ Planned  |
| Quantile uncertainty estimation          | ⏳ Planned  |
| Conformal calibration                    | ⏳ Planned  |
| LSTM / Transformer modeling              | ⏳ Planned  |
| River-network GNN                        | ⏳ Planned  |
| Probabilistic extreme-event head         | ⏳ Planned  |
| Model ablation study                     | ⏳ Planned  |

---

# Technology stack

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/pandas-data%20engineering-150458?logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/NumPy-numerical%20computing-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-machine%20learning-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-residual%20baseline-EB5E28" alt="XGBoost">
  <img src="https://img.shields.io/badge/hydroeval-hydrologic%20metrics-0077B6" alt="hydroeval">
  <img src="https://img.shields.io/badge/Matplotlib-visualization-11557C" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Ruff-code%20quality-D7FF64?logo=ruff&logoColor=black" alt="Ruff">
</p>

---

# Citation

A formal software citation and versioned release will be added when the first stable research release is published.

Until then, please cite:

```text
HYDRO-FLOW-AI
https://github.com/mtariqi/HYDRO-FLOW-AI
```

and acknowledge the upstream NWM-ML project when its workflows or source components materially contribute to the analysis.

---

# Contributing

Reproducibility checks, hydrology-domain feedback, feature suggestions, model comparisons, and issue reports are welcome.

For substantial changes, please open an issue describing:

1. the hydrologic or machine-learning problem;
2. the proposed change;
3. expected validation criteria;
4. implications for temporal leakage;
5. implications for extreme-event evaluation;
6. implications for uncertainty or calibration.

---

<p align="center">
  <strong>HYDRO-FLOW-AI</strong><br>
  Better average predictions are useful.<br>
  <strong>Better extreme-event predictions are the objective.</strong>
</p>
