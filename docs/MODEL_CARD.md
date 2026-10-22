# Model card

## Intended use

Research and decision-support experiments for post-processing river-discharge
forecasts, diagnosing station-level model failures, and comparing hydrological
regimes.

## Out-of-scope use

The project must not independently issue public flood warnings, control water
infrastructure, or replace forecasts from authorized agencies.

## Current model

Version 0.2 uses histogram gradient boosting to predict the residual between
observed flow and NWM-like physical forecasts. Corrected predictions are clipped
at zero. Extreme observations receive increased training weight.

## Limitations

- The included data are synthetic.
- Prediction intervals and temporal GNN layers are planned, not yet production-ready.
- Synthetic performance does not establish real-world skill.
- Reservoir releases and diversions may be unavailable or delayed.
- Graph-RAG answers can inherit errors from source documents and extraction.

## Safety controls

- NWM fallback when inputs are missing or stale
- Separate monitoring of strong and weak stations
- Event metrics in addition to aggregate error
- Evidence identifiers in generated diagnostics
- Human review for operational or scientific claims

