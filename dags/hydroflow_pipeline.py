"""Optional Airflow DAG for batch retraining; live inference stays outside Airflow."""

from __future__ import annotations

from datetime import datetime

try:
    from airflow.decorators import dag, task
except ImportError:  # Keeps repository importable without Airflow.
    dag = task = None


if dag is not None:
    @dag(schedule="0 3 * * 1", start_date=datetime(2026, 1, 1), catchup=False, tags=["hydrology"])
    def hydroflow_batch():
        @task.bash
        def validate_and_train():
            return "hydroflow train --data /data/validated.csv --output /artifacts/challenger"

        @task.bash
        def evaluate():
            return "hydroflow evaluate --data /data/validated.csv --artifacts /artifacts/challenger"

        @task.bash
        def update_knowledge():
            return "hydroflow build-knowledge --artifacts /artifacts/challenger"

        update_knowledge.override(trigger_rule="all_success")()(evaluate()(validate_and_train()))

    hydroflow_batch()

