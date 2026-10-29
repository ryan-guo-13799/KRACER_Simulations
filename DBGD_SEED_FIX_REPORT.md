# DBGD policy seed correction

The adapter now initializes DBGD with origin's separate policy seed:

```python
np.random.SeedSequence([sim_config.seed, 0xDB6D, 17])
```

Previously DBGD and the private user each started a generator with the raw
simulation seed. Their first normalized Gaussian vectors were identical,
making DBGD's initial incumbent equal to the hidden true preference.

## Changed files and functions

- `src/recommenders/baseline_support/adapter.py`:
  `PairwiseAdapter.__init__` selects the separate seed when the policy class
  is DBGD. The other baseline constructors keep their existing seeds.
  `BEGIN DBGD SEED FIX` / `END DBGD SEED FIX` comments mark the changed block.
- `tests/test_unified_runner.py`: the new
  `test_dbgd_uses_independent_origin_seed` covers seeds 6000-6019 at dimensions
  3, 5, and 9. It checks the origin initialization and following random draw,
  and that the incumbent differs from the true preference.

The DBGD learning rule, step sizes, user model, other algorithms, scenario
settings, seed counts, and launch scripts are unchanged.

## Verification

- The regression test failed on the old code before applying the fix.
- The complete simulation test suite passes: **188 tests**.
- Twenty actual baseline runs, each 10,000 rounds, match the prior seed-only
  in-memory experiment for both regret metrics to within 1e-10.
- Mean cumulative slate-average regret is **3277.549300**, versus **18.172796**
  in the old saved rewrite run.
- The source fingerprint check confirms that only the baseline adapter changed
  among the 76 source files from both simulation packages.
- Per-seed evidence is in `verification/dbgd_seed_fix/results.json`.

Origin's saved baseline mean is 4076.462643. The separately investigated
round-one floating-point tie differences remain; this change fixes seeding
only. Existing experiment outputs are preserved. New runs use the correction;
the old saved DBGD results must be regenerated for a corrected comparison.

## Undo

The pre-fix commit is `4a61636` (`parallel-progress`). This correction is tagged
`dbgd-seed-fix`. Undo only this correction with:

```powershell
git revert dbgd-seed-fix
```

Unrelated local changes and the earlier untracked investigation material are
excluded from this fix commit.
