# Manual K-RACER ablations

The four ablations share K-RACER's recommendation/update loop and the same user,
startup, confidence schedule and information updates. Added variant code is
marked `EXTRA FOR THIS ABLATION` in each module.

| Variant | Added operations | Shared operations |
| --- | --- | --- |
| `AnalyticalCertificate` | Analytical admission and a check of the actual restart contraction | EQ, Cut, Restart, directional excitation |
| `MinimumEigenvector` | Excitation centered on the least-informed eigenvector of V | Winner tree, excitation ring, Cut, Restart |
| `BasisExcitation` | A frozen eigenbasis and a D-response excitation block | Winner tree, excitation ring, Cut, Restart |
| `WithoutRestart` | Persistent coordinate intervals, an interval-width winner tree and a midpoint estimate | EQ, Cut, SafeAngularBounds, directional excitation |

`KRACER.py` exposes small methods for these changes. `WorstNormal` and
`ExcitationRing` in `KRACER_functions.py` are extracted shared calculations.
The no-restart variant relaxes only Cut's largest-axis check: selection uses
remaining interval widths, while the angular bounds still use the original W.

Use `run_simulation(sim, rec, algorithm="minimum_eigenvector")`. Available names:
`kracer`, `analytical_certificate`, `minimum_eigenvector`, `basis_excitation`,
and `without_restart`. The analytical gate uses `rec.analytical_gate_size`,
which defaults to 4. The ten-case runner accepts `--algorithms` and defaults
to all five methods, saving each method in its own directory.

These use the manual project's ordinary floating-point geometry. They do not
reproduce the original project's outward numerical bounds or parameter search.
The eigenvector and basis variants also retain the manual ring convention;
the original simulation additionally orients excitation caps toward the cone
axis. Those choices can affect trajectories and regret.
