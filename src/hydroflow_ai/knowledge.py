from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    title: str
    text: str
    source: str
    entity_type: str
    entities: tuple[str, ...] = ()


class LocalEvidenceGraph:
    """Deterministic offline graph-retrieval baseline.

    This makes the project testable without an LLM. A production adapter can
    index the same normalized evidence records in RAG-Anything/LightRAG.
    """

    def __init__(self, records: Iterable[Evidence] = ()):
        self.records = list(records)
        self.entity_to_evidence: dict[str, set[int]] = {}
        for idx, record in enumerate(self.records):
            for entity in record.entities:
                self.entity_to_evidence.setdefault(entity.lower(), set()).add(idx)

    def add(self, record: Evidence) -> None:
        idx = len(self.records)
        self.records.append(record)
        for entity in record.entities:
            self.entity_to_evidence.setdefault(entity.lower(), set()).add(idx)

    def query(self, question: str, limit: int = 5) -> list[dict]:
        tokens = set(re.findall(r"[a-z0-9_\-.]+", question.lower()))
        expanded: set[int] = set()
        for token in tokens:
            expanded.update(self.entity_to_evidence.get(token, set()))
        scored = []
        for idx, record in enumerate(self.records):
            haystack = set(re.findall(r"[a-z0-9_\-.]+", f"{record.title} {record.text}".lower()))
            score = len(tokens & haystack) + (2 if idx in expanded else 0)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].evidence_id))
        return [{"score": score, **asdict(record)} for score, record in scored[:limit]]

    def save(self, path: str | Path) -> None:
        payload = [{**asdict(record), "entities": list(record.entities)} for record in self.records]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LocalEvidenceGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Evidence(**{**row, "entities": tuple(row.get("entities", []))}) for row in payload)


def build_evaluation_evidence(evaluation_path: str | Path) -> LocalEvidenceGraph:
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    graph = LocalEvidenceGraph()
    for station in evaluation["stations"]:
        station_id = station["station_id"]
        text = (
            f"Baseline KGE {station['baseline']['kge']:.3f}; corrected KGE "
            f"{station['corrected']['kge']:.3f}; delta KGE {station['delta_kge']:.3f}; "
            f"failure cohort {station['failure_cohort']}."
        )
        graph.add(
            Evidence(
                evidence_id=f"evaluation:{station_id}",
                title=f"Evaluation for {station_id}",
                text=text,
                source=str(evaluation_path),
                entity_type="model_evaluation",
                entities=(station_id, "KGE", "NWM", "residual-correction"),
            )
        )
    return graph

