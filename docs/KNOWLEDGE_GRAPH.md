# Hydrology knowledge graph

## Purpose

The knowledge graph supports scientific retrieval and auditability. It does not
perform hydrologic routing or replace the forecasting model.

## Example path

```text
Gauge S003
  -> mapped_to -> NWM reach 123
  -> located_in -> snow-dominated catchment
  -> evaluated_in -> run abc123
  -> produced -> baseline KGE -0.18
  -> corrected_by -> residual model v0.2
  -> produced -> corrected KGE 0.41
  -> explained_by -> SWE and storage-change evidence
  -> supported_by -> paper/manual/figure citation
```

## Query classes

- Local: “What happened at station S003?”
- Relational: “Which upstream reservoirs affect weak stations?”
- Comparative: “Which snow basins improved after adding SWE?”
- Global: “What mechanisms recur across negative-KGE stations?”
- Provenance: “Which data version and commit produced Figure 4?”

## Evidence requirements

Every answer should identify its evidence records. Numerical claims must be
checked against structured metrics rather than copied from generated prose.
Missing evidence must be reported explicitly. Causal language is prohibited
unless the evidence supports causal identification.

