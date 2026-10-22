from pathlib import Path

import numpy as np

from hydroflow_ai.data import chronological_split, make_synthetic_dataset, validate_dataset
from hydroflow_ai.diagnostics import diagnose_station
from hydroflow_ai.evaluate import evaluate_model
from hydroflow_ai.knowledge import Evidence, LocalEvidenceGraph, build_evaluation_evidence
from hydroflow_ai.metrics import hydrologic_metrics
from hydroflow_ai.network import adjacency_from_metadata
from hydroflow_ai.train import train_model


def test_synthetic_data_and_split():
    frame = make_synthetic_dataset(stations=4, periods=120)
    validate_dataset(frame)
    split = chronological_split(frame)
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()


def test_directed_adjacency():
    frame = make_synthetic_dataset(stations=4, periods=20)
    stations, adjacency = adjacency_from_metadata(frame)
    assert stations == ["S000", "S001", "S002", "S003"]
    assert adjacency.shape == (4, 4)
    assert int((adjacency > 0).sum()) == 3
    assert np.allclose(adjacency[:3].sum(axis=1), 1)


def test_perfect_metrics():
    metrics = hydrologic_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert np.isclose(metrics.kge, 1)
    assert np.isclose(metrics.nse, 1)
    assert np.isclose(metrics.rmse, 0)


def test_graph_retrieval():
    graph = LocalEvidenceGraph(
        [Evidence("e1", "Station S003", "Negative baseline KGE", "run.json", "metric", ("S003", "KGE"))]
    )
    result = graph.query("Why is S003 KGE negative?")
    assert result[0]["evidence_id"] == "e1"


def test_end_to_end(tmp_path: Path):
    data = tmp_path / "demo.csv"
    artifacts = tmp_path / "artifacts"
    make_synthetic_dataset(stations=5, periods=240).to_csv(data, index=False)
    train_model(data, artifacts)
    result = evaluate_model(data, artifacts)
    graph = build_evaluation_evidence(artifacts / "evaluation.json")
    graph.save(artifacts / "evidence_graph.json")
    assert result["summary"]["stations"] == 5
    assert len(graph.records) == 5
    assert (artifacts / "test_predictions.csv").exists()
    report = diagnose_station(artifacts, "S000")
    assert "evaluation:S000" in report
    assert "evaluation:S001" not in report

