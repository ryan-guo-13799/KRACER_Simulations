**Remove the global excitation-slate sign flip — 30 October 2026**

The only production change is in
[KRACER_functions.py](src/recommenders/KRACER_functions.py), function
`ExcitationSlate`. Previously it negated the entire slate when its mean
pointed away from the cone axis. It now returns the reference-ring slate
directly, preserving the signed excitation center in the paper's Equations
(43)–(44) and Algorithm 8. The edited block is bracketed by
`BEGIN PAPER ALIGNMENT CHANGE` / `END PAPER ALIGNMENT CHANGE` comments.

The tangent-frame construction, reference ring, worst-normal selection,
radius, startup, and recommendation/update loop are retained. The origin
under `simulations` is unchanged. K-RACER, analytical certificate, and without
restart inherit this shared helper change. The minimum-eigenvector ablation
calls the ring directly, so its excitation method is unchanged. The broader
ablation fixes remain pending.

The test that required a forward-facing cap now requires the original signed
cap. Four regression cases that required the oriented origin's full-run totals
were replaced with the paper's signed-cap conditions for K=2,3,5,9. They check
that items lie around the prescribed signed center, with unit perpendicular
perturbations, including a symmetric-query case whose center points away from
the cone axis. Historic origin-comparison data were not overwritten.

Validation: **98 tests passed**. K-RACER and the three available ablations
each completed a 10,000-response run at D=5, K=2, normalized noise=0.10,
seed=6000, with finite regrets and matching public user/recommender information
matrices. Settings and results are recorded in
[smoke_results.json](verification/paper_orientation/smoke_results.json).
`git diff --check` passed for the edited source and test.

The checkpoint is commit `cef7824`, tagged `kracer-before-no-global-flip`.
This removal is a separate commit tagged `kracer-no-global-flip`. From
`manual_simulations`, undo just this removal with:

```powershell
git revert kracer-no-global-flip
```

Unrelated working-tree changes were excluded from this commit.
