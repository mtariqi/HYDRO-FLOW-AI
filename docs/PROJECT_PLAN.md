# Research and implementation plan

## Objective

Develop a physics-guided post-processing system that improves NWM discharge
predictions at weak stations while preserving performance at reliable stations.
Add an evidence-grounded assistant for diagnosis, reproducibility, and research
synthesis.

## Primary hypothesis

Regime-aware residual learning that uses gauge history, meteorology, snow,
reservoir, catchment, and upstream-network information will improve KGE, bias,
peak timing, and extreme-flow detection at stations with baseline KGE <= 0.

## Delivery phases

### Phase 0 — synthetic, reproducible baseline

- Data contracts and validation
- Chronological splitting
- NWM persistence baseline
- Gradient-boosted residual corrector
- Station-stratified KGE and event metrics
- Offline evidence graph and diagnostic reports

### Phase 1 — real pilot

- Select 20–30 gauges spanning snow, rain, regulated, and natural regimes
- Ingest issue-time-valid NWM forecasts and USGS observations
- Add NOAA weather, SNOTEL, reservoir, and catchment features
- Implement event-based splits and forecast-horizon evaluation

### Phase 2 — spatial-temporal modelling

- Temporal encoder (GRU/TFT)
- Directed reach graph with travel-time-aware message passing
- Quantile forecasts and calibrated exceedance probabilities
- Adaptive gating among NWM, tabular correction, and graph model

### Phase 3 — multimodal RAG

- Ingest technical manuals, papers, tables, maps, equations, and figures
- Link station metadata, model runs, metrics, and modelling decisions
- Require citations and numerical verification
- Benchmark retrieval and unsupported-claim rates

### Phase 4 — MLOps and monitoring

- Champion/challenger registry
- Data freshness, drift, calibration, and degradation monitoring
- NWM fallback policy
- Cloud deployment and scheduled batch retraining

## Ablations

| Experiment | Removed component | Question |
|---|---|---|
| A1 | Reservoir features | Do operations explain regulated-basin error? |
| A2 | SWE/snow features | Are gains concentrated in snow-dominated basins? |
| A3 | Upstream features | Does network context improve timing? |
| A4 | Recent residuals | How much gain comes from live assimilation? |
| A5 | Event weighting | Does extreme detection improve without average-skill loss? |
| A6 | Physics residual | Is correction safer than direct flow prediction? |
| A7 | Graph-RAG | Does connected retrieval improve diagnostic accuracy? |

## Promotion criteria

A challenger is promoted only if it improves median station KGE, moves a
meaningful fraction of the negative-KGE cohort above zero, does not materially
reduce flood CSI, preserves uncertainty calibration, and does not degrade more
than a predefined fraction of strong stations.

