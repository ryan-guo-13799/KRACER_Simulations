# Full parallel experiment

From `manual_simulations`, run in Git Bash on Windows or Bash on Linux:

```bash
bash run_all.bash
```

The launcher starts one `simulation_main.py` process per algorithm, up to eight
concurrently. Each process handles its own scenarios and seeds. Use
`JOBS=4 bash run_all.bash` to limit concurrency, or set `PYTHON=python3` if that
is the executable in your environment. `run.bash` remains an alias.

## Default experiment size

| Quantity | Default |
| --- | --- |
| Algorithms | 8: K-RACER, three ablations, RAES, DBGD, Doubler, Sparring |
| Scenarios | 14 total; 5 have K=2 |
| Test seeds | 20 per eligible algorithm/scenario: 6000-6019 |
| Validation seeds | 4 separate seeds: 120000-120003 |
| Horizon | 10,000 rounds per test run |
| Saved test runs | `(4 * 14 + 4 * 5) * 20 = 1,520` |
| Logical validation evaluations | `4 * 14 * 11 * 3 * 4 + 5 * 11 * 4 = 7,612` |
| Total logical evaluations | 9,132, including the 1,520 test runs |

The five K=2 scenarios are the default case, dimensions 3 and 9, and normalized
noise 0.05 and 0.15. The other nine combine K in {3, 5, 9} with normalized
noise in {0.05, 0.10, 0.15}, at dimension 5. The default case is dimension 5,
K=2, normalized noise 0.10. Normalized noise means `sigma_u * sqrt(dimension)`.

Seeds are reused across methods and scenarios; they identify the random
realizations used for repeat runs. Validation seeds select T0 and parameters;
test seeds evaluate the selected policy. Validation uses the existing forked
search, so logical candidates do not imply independent full-horizon runs.

## Changes and output

Only the Bash launcher and documentation change. Python simulation code,
algorithm implementations, parameter grids, and seed defaults are unchanged.
`BEGIN PARALLEL RUN` / `END PARALLEL RUN` comments bracket the launcher changes.

Each process gets a separate work directory and log. After all finish, the
launcher moves their disjoint outputs into a single
`output/parallel_<unique ID>/results/<scenario>/<algorithm>/` tree. Logs are
under the sibling `logs/` directory. It never overwrites previous experiments.
A failed worker produces a nonzero launcher exit after the other methods
finish, with partial results and logs retained. BLAS threads are limited to
one per process.

Override `JOBS`, `PYTHON`, `HORIZON`, `SEED_COUNT`, `START_SEED`,
`VALIDATION_SEED_COUNT`, `VALIDATION_START_SEED`, or `OUTPUT_ROOT` as needed.
Other main-runner options pass through; the launcher owns `--algorithms` and
`--output`.

## Verification and undo

- Git Bash syntax check passed.
- Two-process smoke run: horizon 12, cutoffs 0 and 2, one policy value and one
  validation/test seed. All 14 scenarios and 76 eligible method/scenario groups
  completed: 198 logical evaluations, zero failures.
- All 137 saved JSON files exactly match the earlier sequential smoke run.
  The plot loader reads the combined directory without issues.
- Eight-process failure check through `run.bash`: overlapping seed ranges
  are rejected by all eight processes; all logs survive; the launcher exits 1.
- Temporary checks are isolated under `verification/parallel_runner/tmp/`.
  The full 10,000-round campaign has not been started.

The pre-change state is tagged `project-cleanup`. The launcher change is tagged
`parallel-runner`; undo it with `git revert parallel-runner`.
