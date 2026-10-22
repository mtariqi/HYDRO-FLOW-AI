from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from .data import make_synthetic_dataset
from .diagnostics import diagnose_station
from .evaluate import evaluate_model
from .knowledge import build_evaluation_evidence
from .train import train_model


app = typer.Typer(help="Physics-guided streamflow correction and graph-RAG diagnostics")


@app.command("make-demo")
def make_demo(output: Path = Path("data/demo.csv"), stations: int = 12, periods: int = 1600):
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = make_synthetic_dataset(stations=stations, periods=periods)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame):,} rows to {output}")


@app.command()
def train(data: Path = Path("data/demo.csv"), output: Path = Path("artifacts")):
    typer.echo(json.dumps(train_model(data, output), indent=2))


@app.command()
def evaluate(data: Path = Path("data/demo.csv"), artifacts: Path = Path("artifacts")):
    typer.echo(json.dumps(evaluate_model(data, artifacts)["summary"], indent=2))


@app.command("build-knowledge")
def build_knowledge(artifacts: Path = Path("artifacts")):
    graph = build_evaluation_evidence(artifacts / "evaluation.json")
    graph.save(artifacts / "evidence_graph.json")
    typer.echo(f"Indexed {len(graph.records)} evidence records")


@app.command()
def diagnose(artifacts: Path = Path("artifacts"), station: str = "S000"):
    typer.echo(diagnose_station(artifacts, station))


@app.command()
def smoke(workdir: Path = Path("smoke_output")):
    data = workdir / "demo.csv"
    artifacts = workdir / "artifacts"
    workdir.mkdir(parents=True, exist_ok=True)
    make_synthetic_dataset(stations=8, periods=600).to_csv(data, index=False)
    train_model(data, artifacts)
    result = evaluate_model(data, artifacts)
    graph = build_evaluation_evidence(artifacts / "evaluation.json")
    graph.save(artifacts / "evidence_graph.json")
    typer.echo(json.dumps(result["summary"], indent=2))


@app.command()
def serve(artifacts: Path = Path("artifacts"), host: str = "0.0.0.0", port: int = 8000):
    os.environ["HYDROFLOW_ARTIFACTS"] = str(artifacts.resolve())
    import uvicorn

    uvicorn.run("hydroflow_ai.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()

