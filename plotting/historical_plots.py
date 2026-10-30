"""Historical simulation outcomes, without simulation imports.

The historical campaign is kept separate from the sphere experiments.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import METRICS, STYLE, error_half, mean_ci, save_figure
from .data import MissingDataError


def render_historical(study, output):
    """Plot Fig. 24 only if its cases exist in the supplied manual saved files."""
    if study.get("historical_records"):
        figures = _render_historical_records({"records":study["historical_records"]},output)
        stops = sum(r["summary"].get("precision_stop") is not None for r in study["historical_records"])
        figures[0]["description"] = ("Fresh supplied-cone runs of the historical protocol with the current manual implementation: D=3, a 45-degree cone, T0=64, contraction 0.9, normalized noise 0.10, and disjoint paired seeds 26000-26019. Perfect users choose by true utility with zero comparison margins; learning users use Bayesian-CARA scores. These are new runs, not the paper's archived numerical outcomes. "
            +f"Numerical-precision stops: {stops}. Connecting lines do not fit scaling laws.")
        return figures
    names = {f"cone_K{k}_{model}" for k in (2,3,5) for model in ("perfect","bayesian")}
    names.update({"cone_risk_-5","cone_risk_0","cone_risk_5"})
    missing = sorted(names-set(study["cases"]))
    if missing:
        raise MissingDataError("Historical supplied-cone runs absent from manual input: " + ", ".join(missing))
    records = []
    for cid in names:
        case = study["cases"][cid]
        if len(case["keys"]) != 1:
            raise MissingDataError(f"Historical case {cid} must identify one evaluated policy")
        for summary in case["raw_selected"][case["keys"][0]]["summaries"]:
            records.append({"case":cid,"summary":summary})
    return _render_historical_records({"records":records},output)


def _interval_points(ax, x, values, label, color, marker="o"):
    """Draw mean and two-sided marginal intervals from independent seed values."""
    stats = [mean_ci(v) for v in values]
    means = np.asarray([s["mean"] for s in stats])
    lo = np.asarray([error_half(s) for s in stats])
    hi = np.asarray([s["hi"] - s["mean"] if s["hi"] is not None else 0. for s in stats])
    ax.errorbar(x, means, yerr=np.vstack([lo, hi]), color=color, marker=marker,
                capsize=3, linewidth=1.4, markersize=4, label=label)


def _render_historical_records(payload: dict, output: Path) -> list[dict]:
    """Recreate the three-panel supplied-cone figure from its saved raw records.

    Accept either the original complete JSON payload or its manuscript extract
    containing ``protocol`` and ``records``.  The summary schema is identical.
    Twenty original seed outcomes remain the observational unit; no current
    sphere outcome is pooled with them.
    """
    groups = defaultdict(list)
    for record in payload["records"]:
        groups[record["case"]].append(record)
    expected = {f"cone_K{k}_{model}" for k in (2, 3, 5)
                for model in ("perfect", "bayesian")}
    expected.update({"cone_risk_-5", "cone_risk_0", "cone_risk_5"})
    missing = sorted(expected.difference(groups))
    if missing:
        raise ValueError("Historical figure requires cases: " + ", ".join(missing))

    consumed = METRICS[0]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.65), layout="constrained")
        for model, label, color in (("perfect", "Perfect user", "#a54e2f"),
                                    ("bayesian", "Learning user", "#3267a8")):
            values = [[r["summary"][consumed] for r in groups[f"cone_K{k}_{model}"]]
                      for k in (2, 3, 5)]
            _interval_points(axes[0], [2, 3, 5], values, label, color)
        axes[0].set(title="(a) Perfect and learning users", xlabel="Maximum slate size K",
                    ylabel="Consumed-item regret", yscale="log", xticks=[2, 3, 5])
        axes[0].legend(fontsize=8)

        for phase, label, color in (("excite", "Excitation", "#c17d58"),
                                     ("localize", "Localization", "#328f89")):
            counts = [np.mean([r["summary"]["phase_counts"].get(phase, 0)
                               for r in groups[f"cone_K{k}_bayesian"]]) for k in (2, 3, 5)]
            axes[1].plot([2, 3, 5], counts, marker="o", color=color, label=label)
        axes[1].set(title="(b) Learning-user exploration", xlabel="Maximum slate size K",
                    ylabel="Mean epochs within fixed $T_0=64$", xticks=[2, 3, 5], ylim=(0, 64))
        axes[1].legend(fontsize=8)

        names = ["cone_risk_-5", "cone_risk_0", "cone_K2_bayesian", "cone_risk_5"]
        _interval_points(axes[2], [-5, 0, 1, 5],
                         [[r["summary"][consumed] for r in groups[name]] for name in names],
                         "Learning user", "#3267a8", marker="s")
        axes[2].set(title="(c) Signed risk attitudes, K=2", xlabel="CARA risk coefficient",
                    ylabel="Consumed-item regret", xticks=[-5, 0, 1, 5])
        for ax in axes:
            ax.grid(alpha=.2)
        artifact = save_figure(fig, output, "historical_supplied_cone",
                               "Historical supplied-cone results", number=24)
    return [artifact]

