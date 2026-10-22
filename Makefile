.PHONY: install demo train evaluate knowledge smoke workflow-dryrun workflow test lint

install:
	pip install -e ".[dev,api,workflow]"

demo:
	hydroflow make-demo --output data/demo.csv

train:
	hydroflow train --data data/demo.csv --output artifacts

evaluate:
	hydroflow evaluate --data data/demo.csv --artifacts artifacts

knowledge:
	hydroflow build-knowledge --artifacts artifacts

smoke:
	hydroflow smoke

workflow-dryrun:
	snakemake --snakefile workflow/Snakefile --workflow-profile workflow/profiles/local --dry-run

workflow:
	snakemake --snakefile workflow/Snakefile --workflow-profile workflow/profiles/local

test:
	pytest -q

lint:
	ruff check .

