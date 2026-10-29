# Unified simulation runner cleanup

Run from `manual_simulations`:

```powershell
python src/simulation_main.py
```

This runs K-RACER, analytical-certificate, minimum-eigenvector, without-restart,
RAES, DBGD, Doubler, and Sparring in one experiment directory. Methods execute
sequentially. The four pairwise baselines participate only in K=2 cases.
`--algorithms` selects a subset.

## File changes

The six runner modules are reduced to three:

| Remaining module | Responsibility |
| --- | --- |
| `src/simulation_main.py` | Cases, CLI, validation selection, held-out runs, JSON output for all eight methods |
| `src/simulation_state.py` | Shared interaction loop, forkable state, `run_simulation` |
| `src/cutoff_search.py` | Safe prefix sharing, cutoff scores, validation selection |

- Removed `src/run_pairwise_baselines.py`; main now schedules those methods.
- Removed `src/cutoff_validation.py`; its selection functions moved into cutoff search.
- Removed `src/simulation_function.py`; its thin wrapper moved into simulation state.
- Updated imports in tests and verification scripts.
- Both Bash launchers now invoke the unified main runner. `run.bash` is a thin
  alias for `run_all.bash`. Bash files retain LF endings through `.gitattributes`.
- Updated README usage. Changed implementation blocks have
  `BEGIN RUNNER CLEANUP` / `END RUNNER CLEANUP` comments.

## Preserved behavior

No recommender algorithm implementation was changed by this cleanup. K-RACER
and its three ablations still use coordinate-sign startup, post-startup T0,
and their existing parameter grids. RAES starts immediately and searches an
absolute T0. DBGD, Doubler, and Sparring start immediately and learn throughout
the horizon without cutoff selection.

Forked T0 search remains in place and now also handles RAES. Copies retain
private user histories and RNG states. RAES validation keeps the old baseline
runner's sequential floating-point accumulation to preserve exact scores.

Study cases, default grids, validation/test seed separation, trajectories,
regret definitions, and saved round fields remain the same. The only deliberate
record-value adjustment is the baseline's display-only reference angle: it now
uses cone startup's rounding so the existing plot loader accepts mixed-method
cases. At D=5 this differs by one floating-point unit (about 2.22e-16 radians).
Baselines do not use this angle in their calculations.

Historic experiment outputs were not overwritten. Pre-existing plotting,
documentation, ignore-file, and recommender-comment edits remain outside these
commits. Existing untracked verification work remains untracked; its import of
the removed simulation wrapper was retargeted to simulation state.

## Verification

- `tests`: **128 passed**. New coverage checks baseline startup exclusion,
  K=2 restrictions, RAES forks and private RNG state, all-eight default CLI
  scheduling, metadata, and compatibility with the existing plot loader.
- `verification/unified_runner/verify_cleanup.py` loads the pre-cleanup
  runners and cone simulation engine from Git. At D=3 and D=5 it verifies
  **8 baseline records** and **8 cone records**, including exact trajectories
  and regret. Complete cone records match exactly; baseline records match
  after the display-only angle normalization described above.
- **12 RAES cutoff scores** match the old independent baseline loop exactly.
  Evidence and current source hashes are in
  `verification/unified_runner/results.json`.
- Actual CLI check: horizon 12, cutoffs 0 and 2, one validation seed and one
  held-out seed, across all **14 cases**. **198 logical evaluations completed,
  zero failed or skipped**, producing 76 case/method groups. The plot loader
  reads all groups with no issues. Temporary results are isolated under
  `verification/unified_runner/tmp/cli_smoke`.
- Git Bash syntax checks pass for both launch scripts.

The full default 10,000-round campaign was not rerun for this cleanup.

## Git checkpoints and undo

- `unified-before-cleanup`: state before this task (`adc0c85`).
- `cutoff-search-cleanup`: cutoff-module merge (`2d87090`).
- `project-cleanup`: unified runner and remaining cleanup.

To undo both cleanup commits while retaining history:

```powershell
git revert project-cleanup cutoff-search-cleanup
```

The pre-existing untracked ablation comparison script is outside these
commits. If restoring the removed wrapper, its import can likewise be changed
back from `simulation_state` to `simulation_function`.
