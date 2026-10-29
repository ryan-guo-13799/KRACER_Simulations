**K-RACER parity fix, step 1 — 30 October 2026**

This step makes the rewrite's K-RACER results substantially closer to the
origin while keeping its recommendation/update loop, state structure, user
model, cuts, winner tree, and restart implementation. Only three production
source files were edited. This is a close-results fix, not a claim of exact
trajectory reproduction.

Before any simulation code was changed, the existing Git repository was
checkpointed at **`f999b53c61737e001f66057dda4ebd216393b547`**, named
`Checkpoint rewrite before K-RACER parity changes`. It is also tagged
**`kracer-before-step1`**. The old repository tracked only README.md; this
checkpoint additionally records the then-current rewrite source, tests,
plotting scripts and run scripts. Generated simulation/plot outputs are left
on disk and are not included in Git. The fix is a separate commit tagged
**`kracer-step1`**.

Every edited production block has comments before and after it, containing
`BEGIN K-RACER PARITY CHANGE (step 1)` and
`END K-RACER PARITY CHANGE (step 1)`.

The changes are:

1. [startup_functions.py](src/recommenders/startup_functions.py): replaced
   the QR basis in `make_cone` with canonical-axis projection, using a small
   shared `TangentFrame` helper. Both bases describe the same cap, but basis
   order determines which direction the policy selects when widths are tied.
   The new convention matches the origin's `geometry.tangent_frame`.

2. [KRACER_functions.py](src/recommenders/KRACER_functions.py): matched the
   origin's ordering and floating-point calculation of candidate worst-normal
   angles; added an optional reference argument to `ExcitationRing`; used
   the public cone axis as that reference for directional excitation; and
   flipped an entire excitation slate when its mean points away from the cone
   axis. The reference-ring path includes the origin's fallback for a nearly
   parallel reference. The old no-reference ring path is retained for the
   minimum-eigenvector ablation. The helper implementations are local to the
   rewrite; running K-RACER does not import or require the origin project.

3. [startup.py](src/recommenders/startup.py): matched the origin's default
   angle calculation: take the midpoint in degrees, then convert to radians.
   The old mathematically equivalent radian calculation differs by one
   floating-point step (2.22e-16 radians at D=5). A regression test showed
   that this tiny difference can later change a near-zero orientation decision.
   Explicit user-supplied startup angles are still used as supplied.

The core [KRACER.py](src/recommenders/KRACER.py) file was not edited. No
ablation-specific class, baseline, experiment grid, cutoff search, or metric
definition was changed. The shared startup and directional-excitation helpers
are also used by ablations, so their outputs may change as an inherited effect;
their separate parity fixes and evaluation remain for later steps.

Additional files provide seven regression tests, the comparison script/results,
this report, and a `.gitignore` for generated outputs, bytecode, editor settings,
and temporary test data. Concurrent changes to plotting files were left in the
working tree and excluded from the fix commit.

The baseline uses D=5, K=2, normalized noise=0.10, horizon=10,000, gamma=0.95,
post-startup T0=64, and all twenty test seeds 6000–6019. Mean final
slate-average regret is:

| Version | Mean regret | Absolute gap from origin |
|---|---:|---:|
| Origin | 97.462582809 | — |
| Rewrite before this step | 98.295673022 | 0.833090213 |
| Rewrite after this step | 97.543565860 | 0.080983052 |

This reduces the mean gap by approximately **90.3%**. The remaining gap is
**0.0831%** of the origin mean. The before value comes from the preserved
rewrite output `output/parallel_qF10dNsc/kracer/baseline/kracer`, with the same
selected parameters and test seeds. The origin's fresh baseline mean matches
its saved `sphere_slow_baseline_20` mean.

For baseline seed 6000 specifically, final slate-average regrets are
99.5057664262 (origin), 95.4303770413 (before), and 99.5057664356 (after).
Seed 6001 now gives 87.6198457532 versus the origin's 87.6198457477.

The broader comparison uses the origin's selected gamma and T0 for each case,
all twenty baseline seeds, and seeds 6000/6001 for each of the other nine cases.
Every run has 10,000 responses. The comparison runs the edited source directly;
it does not substitute functions in memory.

| Case | Seeds | Origin mean slate regret | Rewrite mean slate regret | Relative mean gap |
|---|---:|---:|---:|---:|
| Baseline | 20 | 97.462583 | 97.543566 | 0.083091% |
| D=3 | 2 | 32.951671 | 32.951671 | <0.000001% |
| D=9 | 2 | 513.021811 | 513.022156 | 0.000067% |
| Noise=0.05 | 2 | 34.914494 | 34.914494 | <0.000001% |
| Noise=0.20 | 2 | 417.858813 | 419.166626 | 0.312980% |
| K=3 | 2 | 103.092647 | 103.223573 | 0.126998% |
| K=5 | 2 | 96.260215 | 96.260215 | <0.000001% |
| K=9 | 2 | 84.193555 | 84.193555 | <0.000001% |
| Startup target fraction=0.25 | 2 | 296.122721 | 296.122721 | <0.000001% |
| Startup target fraction=0.75 | 2 | 373.974408 | 374.071012 | 0.025832% |

The 38-run data are in
[results.json](verification/kracer_step1/results.json). This broad run supplied
the origin's startup angles explicitly. Its recorded source fingerprint
therefore predates the final default-angle rounding fix. The new default
calculation produces those same angles. A subsequent ordinary-default check
for baseline seeds 6000/6001 passes the strict tolerances and records the final
source hashes in
[results_default_startup.json](verification/kracer_step1/results_default_startup.json).
The final regression tests also use the ordinary default startup path.

**Residual differences remain.** Twenty-five of the 38 broad runs match
displayed actions within 1e-7, winners and normalized phases exactly, and both
cumulative-regret curves within 1e-4. The other thirteen fail this stricter
trajectory criterion. The maximum absolute difference between cumulative
regret curves anywhere in the broad check is 2.836359; the largest paired final
slate-regret gap is about 2.615627. The largest baseline final slate gap is
1.145756. Thus the comparison script intentionally exits with status 1 for
the broad run: its strict `all_close` criterion is **false**, even though the
case means are much closer. The recorded tolerances have not been relaxed.

The origin's conservative numerical cuts/restarts and the rewrite's ordinary
arithmetic still differ. Near-zero slate-axis dot products can turn small
geometric differences into different orientation choices. Exact equality would
require a later numerical-parity step. The current runner's parameter grid and
case list also still differ from the origin's; this report compares matched
policies, not newly retuned complete studies. The two-seed checks for additional
cases are limited evidence, not a twenty-seed validation of every case.

Validation: **98 tests pass** (the existing 91 plus seven new regression cases).
The new tests protect basis order, reference-ring construction, forward
orientation, and four representative full-horizon origin regret values.
`git diff --check` passes for the changed source. Checks use Python 3.14.0,
NumPy 2.3.3, SciPy 1.16.2, and one numerical thread. Original saved outputs were
not overwritten.

To rerun, from `manual_simulations` in PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
New-Item -ItemType Directory -Force verification/kracer_step1/tmp | Out-Null
python -B -m pytest -p no:cacheprovider --basetemp=verification/kracer_step1/tmp/pytest-rerun tests -q
python -B verification/kracer_step1/compare_origin.py --output verification/kracer_step1/rerun.json
```

The comparison requires the sibling `simulations` folder and its saved
`sphere_slow_baseline_20/selected_T0.csv`. The production rewrite has no such
dependency. A broad rerun can still return status 1 for the residual strict
trajectory differences described above.

To inspect or undo just this step, from `manual_simulations`:

```powershell
git diff kracer-before-step1 kracer-step1 -- src tests/test_kracer_parity_step1.py
git revert kracer-step1
```

`git revert` creates an undo commit and preserves the checkpoint and history.
The origin repository was not edited.
