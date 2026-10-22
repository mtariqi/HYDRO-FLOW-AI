from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

from .knowledge import LocalEvidenceGraph


def diagnose_station(artifacts_dir: str | Path, station_id: str) -> str:
    graph_path = Path(artifacts_dir) / "evidence_graph.json"
    if not graph_path.exists():
        raise FileNotFoundError("Run `hydroflow build-knowledge` first")
    graph = LocalEvidenceGraph.load(graph_path)
    # Station diagnosis is an entity-constrained retrieval task. A broad hybrid
    # query can rank other stations because they share terms such as KGE and
    # NWM, so require an exact station entity before composing the report.
    evidence = [
        {"score": 1, **asdict(record)}
        for record in graph.records
        if station_id.lower() in {entity.lower() for entity in record.entities}
    ]
    if not evidence:
        return f"No indexed evidence was found for {station_id}."
    lines = [f"Station diagnostic: {station_id}", ""]
    for item in evidence:
        lines.append(f"- {item['text']} [{item['evidence_id']}; source: {item['source']}]")
    lines.extend(
        [
            "",
            "Interpretation guardrail: these statements summarize indexed results; they do not "
            "establish causal hydrologic mechanisms without independent evidence.",
        ]
    )
    return "\n".join(lines)

