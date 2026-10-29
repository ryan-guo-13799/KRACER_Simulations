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
every case. Methods are evaluated sequentially within this one run.

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

`bash run_all.bash` and `bash run.bash` launch the same unified runner.

Run tests with:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
New-Item -ItemType Directory -Force verification/unified_runner/tmp | Out-Null
python -B -m pytest -p no:cacheprovider --basetemp=verification/unified_runner/tmp/pytest-rerun tests -q
```
