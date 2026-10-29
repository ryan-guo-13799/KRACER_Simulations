# Plot the saved manual simulations

This folder reads this project's native `seed_*.json` and `validation.json` files. It does not import `src`, run simulations, read the sibling `simulations` project, or change existing code/data. Only the empirical plots in the paper's simulation section and simulation appendix are included: Figures 10-13 and 16-24. There are no geometric illustrations or additional report plots.

From **manual_simulations**:

```powershell
python -B -m plotting --input output/parallel_qF10dNsc
```

Or use `python -B -m plotting` to select the newest saved run under `output/`. Open `plotting/results/index.html`. Figures are PDF, SVG, and PNG, with CSV files under `plotting/results/data/`. The default destination is entirely inside this new plotting folder. Install dependencies if necessary with `python -m pip install -r plotting/requirements.txt`.

```powershell
python -B -m plotting --input output/parallel_qF10dNsc --output plotting/results/another_report
python -B -m plotting --input output/parallel_qF10dNsc --trajectory-stride 1
python -B -m pytest -p no:cacheprovider plotting/tests
```

The loader recursively supports both serial `<run>/<case>/<algorithm>/seed_*.json` and parallel `<run>/<worker>/<case>/<algorithm>/seed_*.json` layouts. Supply one campaign, rather than merging different runs. It reads each file once, extracts losses, and releases the raw slates. The first 128 trajectory points, every tenth subsequent point, and the final epoch are retained by default; `--trajectory-stride 1` retains every epoch. Means and marginal two-sided 95% Student-t intervals come from the available held-out seeds. A single seed has an unavailable interval (blank CSV bounds).

`load_run()` checks round numbering, horizon length, final regret sums, phase counts, saved validation scores and selection, selected test settings, seed identities, and matched true preferences. Incomplete/failed groups are excluded and recorded in `input_issues.csv`; `--include-incomplete` opts into successful partial records with the actual seed count. Paired comparisons require matching seed sets; they are never silently paired by position or shortened to an intersection.

## Saved-data limits

The current manual campaign has eight scenarios, four methods, and slower-user noise **0.15**. These actual settings are used, rather than relabeling them as the paper's ten-case study or noise 0.20. Missing K/noise grid cells and methods remain N/A. The Startup sensitivity panel states when no target sweep exists.

The native format saves **validation** outcomes across budgets and parameters, but saves **test** outcomes only at the selected budget. Figures 20 and 22 therefore show the observed selected-budget test points and explicitly state that test curves were not saved. Validation scores are exported separately in `validation_scores.csv`; they are never presented as held-out test curves. The old stores are never used to fill gaps.

Figure 24 requires the historical supplied-cone/perfect-user/signed-risk cases, which are absent from the current manual run. Its function is retained and reports the missing cases; the CLI records it as unavailable. No historical data is imported from another project. `figure_coverage.csv` records every requested figure's status.

## Python functions

- `data.load_run(path, include_incomplete=False, trajectory_stride=10)` reads the actual manual saved-data format into an in-memory plotting view.
- `data.export_csvs(study, output)` writes endpoint summaries, raw endpoints, phase means/counts, paired differences, configurations, validation scores, trajectories, and input issues.
- `main_plots.render_main(study, output)` draws the baseline, trajectories, all-method comparisons, sensitivity, paired ablations, selected-budget costs, and phase costs.
- `grid_plots.render_grid(data.grid_view(study), output)` draws selected parameters, K effects, selected-budget costs, and phase costs; each `plot_grid_*` function is independently callable.
- `historical_plots.render_historical(study, output)` draws the historical simulation figure when the required cases actually exist in the supplied manual data.

`manifest.json` records the exact saved-file hashes, source hashes, figure coverage, software versions, and output hashes. No simulation implementation is copied or changed.
