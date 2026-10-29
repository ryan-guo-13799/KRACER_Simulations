**Forked T0 search — 30 October 2026**

The main validation runner now shares safe simulation prefixes across T0
candidates for each seed and parameter. The manually written K-RACER and
three ablation classes are unchanged. The signed excitation behavior requested
in the previous change is retained.

The implementation is divided into small modules:

| File | Responsibility |
|---|---|
| [simulation_state.py](src/simulation_state.py) | Own the user, RNG, startup, recommender and history; provide `step`, `fork`, and `finish`. |
| [simulation_function.py](src/simulation_function.py) | Preserve the existing `run_simulation` entry point and result/registry imports. Independent runs use the same step engine. |
| [cutoff_search.py](src/cutoff_search.py) | Decide when to fork and return final validation slate regret for each candidate T0. |
| [simulation_main.py](src/simulation_main.py) | Create case configurations, cache one grid per validation seed/parameter, and save the existing validation/test JSON format. |

The largest T0 owns the live trajectory. All budgets share startup. Before
each later response, a smaller budget forks if its remaining time could
restrict the next localization depth. A copy of the recommender prepares the
next interval and reads its maximum depth; the live policy is not mutated by
this check. During an already accepted localization round, sharing is allowed
only while the remaining responses fit. These checks use the existing policy
hooks, so no new search methods are added to the algorithm classes.

Forking uses `deepcopy` of the whole simulation state. This copies posterior
mean/covariance, public matrices, cone or interval state, history and NumPy RNG
state together. Branches do not share mutable user histories. An analytical
certificate's earlier policy stop is preserved when changing its T0. Unknown
policy classes take a conservative path that forks after startup and executes
ordinary interactions.

Validation only needs a scalar score. Once a known policy starts repeating its
fixed exploitation estimate, the evaluator fills the remaining loss sequence
without simulating irrelevant private consumption. It uses the same ordered
losses and native Python `sum` as the independent JSON runner, preserving
compensated float summation on newer Python versions. Ordinary held-out runs
still execute and record every interaction, including the private updates.

`cutoff_validation.py`, the grids, seed sets, selection objective and tie rules
are unchanged. Each validation seed/parameter calls the forked grid once;
subsequent cutoff callbacks read that grid's cached scalar values. Progress
counts logical candidate evaluations, not newly executed trajectories. Held-out
seeds still use `run_case` independently. The separate pairwise-baseline runner
is outside this change and retains its existing search path.

Changed blocks are marked by `BEGIN T0 FORKING CHANGE` and
`END T0 FORKING CHANGE` comments.

**Validation: 117 tests pass.** Added tests cover exact forked versus independent
scores for all four methods at K=2/3, private-state and history isolation,
unchanged RNG streams, forks before the first depth-dependent query, copies
during active localization, analytical early stops, incomplete startup, startup
certification on the final horizon round, and invalid grids. The real main
runner is also checked against independent validation selection: every score,
selected T0/parameter, and held-out JSON record agrees exactly.

The verification script additionally loads the pre-change independent loop
from Git tag `t0-before-forking`. Its interactions, choices, experiences,
posterior arrays and RNG state agree exactly with the extracted step engine
for all four methods at horizons 2 and 180.

A full-horizon benchmark uses D=5, K=2, normalized noise=0.10, validation seed
120000, T=10,000, and T0 candidates
`[0,1,2,4,8,16,32,64,128]`. Analytical uses gate=128 and gamma=0.5; the other
methods use gamma=0.95. Every candidate score matches the pre-change loop
exactly. Wall times on this machine were:

| Method | Independent | Forked | Speedup |
|---|---:|---:|---:|
| K-RACER | 6.51 s | 0.75 s | 8.72x |
| Analytical certificate | 4.66 s | 0.07 s | 70.28x |
| Minimum eigenvector | 8.03 s | 1.46 s | 5.52x |
| Without restart | 6.96 s | 0.64 s | 10.80x |

These timings reflect safe prefix reuse, avoiding repeated startup, and skipping
fixed exploitation suffixes. They describe one seed/parameter grid, not a
promised campaign-wide speedup. The timed independent baseline itself calculates
scores from full interaction results without building JSON records. The main
runner additionally avoids constructing full validation-round JSON records.

Reproduction and source fingerprints are in
[verify_and_benchmark.py](verification/t0_forking/verify_and_benchmark.py) and
[results.json](verification/t0_forking/results.json). From `manual_simulations`,
with the numerical thread counts set to one:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -B -m pytest -p no:cacheprovider --basetemp=verification/t0_forking/tmp/pytest-rerun tests -q
python -B verification/t0_forking/verify_and_benchmark.py
```

The pre-change commit is `c0e42ef`, tagged `t0-before-forking`. The feature is
saved separately with tag `t0-forking`. Undo only this feature from
`manual_simulations` with:

```powershell
git revert t0-forking
```

Existing unrelated edits, including removed comments in algorithm helpers,
plotting changes, and historical report deletions, were left alone and excluded
from this feature commit. The origin repository and saved experiment outputs
were not edited.
