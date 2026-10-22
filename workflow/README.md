# Snakemake execution

Run all commands from the repository root.

## Local workstation

Install the project in the active environment first:

```bash
pip install -e ".[dev,api,workflow]"
snakemake --snakefile workflow/Snakefile --workflow-profile workflow/profiles/local --dry-run
snakemake --snakefile workflow/Snakefile --workflow-profile workflow/profiles/local
```

The current rules call the installed `hydroflow` command, so
`--software-deployment-method apptainer` must not be added unless a container is
assigned to each rule and contains the installed package.

## Slurm cluster

Install the executor plugin, then replace `CHANGE_ME` values in
`workflow/profiles/slurm/config.yaml`:

```bash
pip install snakemake-executor-plugin-slurm
snakemake --snakefile workflow/Snakefile \
  --workflow-profile workflow/profiles/slurm \
  --dry-run
```

Do not use the Slurm profile on a workstation without a configured Slurm
controller and valid account/partition.

## Real data

Change `input_data` in `configs/workflow.yaml` to the validated feature table
and set `generate_demo: false`. The downstream rules will consume that table;
if it is absent, Snakemake will stop instead of silently generating synthetic
data. The table must satisfy the schema validated by
`hydroflow_ai.data.validate_dataset`.

