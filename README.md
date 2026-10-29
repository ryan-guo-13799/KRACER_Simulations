# K-RACER simulations

Run K-RACER, its three ablations, and four baselines from one entry point.
From this folder:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python src/simulation_main.py
```

The default run includes `kracer`, `analytical_certificate`,
`minimum_eigenvector`, `without_restart`, `raes`, `dbgd`, `doubler`, and
`sparring`. Baselines run only on K=2 cases. K-RACER and its ablations run on
every case. When launched directly with Python, methods run sequentially.

K-RACER and its ablations search post-startup T0 and their policy parameter.
RAES searches an absolute T0. DBGD, Doubler, and Sparring use fixed settings
and learn throughout the horizon. Selection uses validation seeds only;
held-out seeds evaluate the selected policy. T0 searches share safe prefixes.

Use `--algorithms` to select a subset, or run a short pipeline check:

```powershell
python src/simulation_main.py --horizon 12 --cutoffs 0 2 --gamma-values 0.9 --gate-sizes 4 --validation-seed-count 1 --seed-count 1
```

Results are saved in a new `output/manual_cases_<timestamp>/` directory, with
one `case/algorithm/` folder per eligible method. `--output` selects a new
directory explicitly. Saved experiments are not overwritten.

The three runner modules are:

- `src/simulation_main.py`: experiment cases, CLI, selection and JSON output.
- `src/simulation_state.py`: the shared interaction loop and forkable state.
- `src/cutoff_search.py`: T0 evaluation and validation selection.

For the full experiment in parallel, use Git Bash on Windows, or Bash on Linux:

```bash
bash run_all.bash
```

This starts one main-runner process per algorithm, up to eight concurrently.
Each process handles that algorithm's scenarios, parameter selection, and test
seeds. Results are collected into `output/parallel_<unique ID>/results/`, with
separate progress/error logs in the sibling `logs/` folder. Each process uses
one BLAS thread. A failed process makes the launcher exit unsuccessfully after
the other methods finish; partial data and logs are retained. `bash run.bash`
is an alias for this launcher.

To use fewer processes or a specific Python executable:

```bash
JOBS=4 PYTHON=python3 bash run_all.bash
```

Defaults are 10,000 rounds, 20 test seeds (6000-6019), and four separate
validation seeds (120000-120003). The same seed ranges are reused across
methods and scenarios. A seed controls the random realization of a run.

There are 14 scenarios; noise below means `sigma_u * sqrt(dimension)`:

| Scenario group | Dimension | K | Noise | Count |
| --- | --- | --- | --- | --- |
| Baseline | 5 | 2 | 0.10 | 1 |
| Dimension changes | 3, 9 | 2 | 0.10 | 2 |
| User-learning speed changes | 5 | 2 | 0.05, 0.15 | 2 |
| Slate-size changes | 5 | 3, 5, 9 | 0.10 | 3 |
| Slate-size/noise combinations | 5 | 3, 5, 9 | 0.05, 0.15 | 6 |

K-RACER and three ablations run on all 14 scenarios. The four baselines run
on the five K=2 scenarios. That gives `(4 * 14 + 4 * 5) * 20 = 1,520` saved
test runs, each with 10,000 rounds. Validation adds 7,612 logical candidate
evaluations: 11 cutoffs and three policy-parameter values on four validation
seeds for each cone method/scenario, and 11 cutoffs on four seeds for RAES.
Safe-prefix sharing makes these cheaper than separate full simulations.

Environment overrides are `JOBS`, `PYTHON`, `HORIZON`, `SEED_COUNT`,
`START_SEED`, `VALIDATION_SEED_COUNT`, `VALIDATION_START_SEED`, and
`OUTPUT_ROOT`. Other main-runner options can be passed through, for example:

```bash
JOBS=2 HORIZON=12 SEED_COUNT=1 VALIDATION_SEED_COUNT=1 bash run_all.bash --cutoffs 0 2 --gamma-values 0.9 --gate-sizes 4
```

Run tests with:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
New-Item -ItemType Directory -Force verification/unified_runner/tmp | Out-Null
python -B -m pytest -p no:cacheprovider --basetemp=verification/unified_runner/tmp/pytest-rerun tests -q
```
