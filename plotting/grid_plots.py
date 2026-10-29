"""K/noise-grid figures drawn exclusively from saved held-out outcomes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch
import numpy as np

from .common import (
    LABELS,
    METRICS,
    PHASE_COLORS,
    PHASE_LABELS,
    STYLE,
    finals,
    mean_ci,
    phase_values,
    save_figure,
)


KEY = "kracer"
NOISE_COLORS = ("#2166ac", "#c17b2f", "#8b2c65", "#386641", "#756bb1")
METRIC_COLORS = ("#2166ac", "#b85d22")
PHASE_ORDER = ("startup", "localize", "excite", "exploit", "pending")


def _cells(study):
    cells = {}
    for case in study["cases"].values():
        if KEY not in case.get("raw_selected", {}):
            continue
        cfg = case["config"]
        coordinate = (round(float(cfg["sigma_u"]) * np.sqrt(cfg["D"]), 12), int(cfg["K"]))
        if coordinate in cells:
            raise ValueError(f"Duplicate K/noise cell {coordinate}: cannot choose a study record")
        cells[coordinate] = case
    if not cells:
        raise ValueError("The grid study contains no saved K-RACER outcomes")
    return cells, sorted({x[0] for x in cells}), sorted({x[1] for x in cells})


def _stats(case, metric):
    return mean_ci(finals(case, KEY, metric))


def _interval(entry):
    if entry["lo"] is None or entry["hi"] is None:
        return None
    return [[entry["mean"] - entry["lo"]], [entry["hi"] - entry["mean"]]]


def _heatmap(ax, values, ks, noises, title, *, logarithmic=False, fmt=".1f", cmap="viridis"):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if logarithmic and positive.size:
        low, high = float(positive.min()), float(positive.max())
        norm = LogNorm(low, high if high > low else low * 1.01)
    else:
        low = float(finite.min()) if finite.size else 0.0
        high = float(finite.max()) if finite.size else 1.0
        norm = Normalize(low, high if high > low else low + 1)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#e7e9ec")
    # Missing entries and exact zeros are explicitly annotated, independently of color.
    colored = np.ma.masked_invalid(values)
    if logarithmic:
        colored = np.ma.masked_less_equal(colored, 0)
    im = ax.imshow(colored, aspect="auto", cmap=palette, norm=norm)
    for row in range(len(noises)):
        for column in range(len(ks)):
            value = values[row, column]
            text = "N/A" if not np.isfinite(value) else format(value, fmt)
            if not np.isfinite(value) or (logarithmic and value == 0):
                color = "#3b4550"
            else:
                rgba = palette(norm(value))
                color = "white" if np.dot(rgba[:3], [0.299, 0.587, 0.114]) < 0.55 else "#18212b"
            ax.text(column, row, text, ha="center", va="center", color=color, fontsize=9)
    ax.set(xticks=range(len(ks)), xticklabels=ks, yticks=range(len(noises)),
           yticklabels=[f"{noise:.2f}" for noise in noises], xlabel="Maximum slate size K",
           ylabel=r"Normalized noise $\sigma_u\sqrt{D}$", title=title)
    ax.tick_params(length=0)
    return im


def plot_grid_selected_parameters(study, output: Path):
    cells, noises, ks = _cells(study)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), constrained_layout=True)
    for ax, name, title, fmt in zip(
        axes, ("T0", "parameter"),
        (r"Validation-selected budget $T_0$", r"Validation-selected contraction $\gamma_{ctr}$"),
        (".0f", ".2f"),
    ):
        values = []
        for noise in noises:
            row = []
            for k in ks:
                case = cells.get((noise, k))
                value = (case["selected_T0"][KEY] if name == "T0" else
                         case.get("selected_parameters", {}).get(KEY, {}).get("value")) if case else None
                row.append(np.nan if value is None else float(value))
            values.append(row)
        im = _heatmap(ax, values, ks, noises, title, fmt=fmt, cmap="cividis")
        fig.colorbar(im, ax=ax, shrink=0.84)
    return save_figure(fig, output, "current_report/08_grid_selected_parameters",
                       "Validation-selected K-RACER settings in each grid cell", number=19)


def plot_grid_k_effects(study, output: Path):
    cells, noises, ks = _cells(study)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        for index, noise in enumerate(noises):
            means = [_stats(cells[(noise, k)], metric)["mean"] if (noise, k) in cells else np.nan
                     for k in ks]
            color = NOISE_COLORS[index % len(NOISE_COLORS)]
            ax.plot(ks, means, marker="o", ms=4, color=color, label=f"Noise {noise:.2f}")
            for k in ks:
                if (noise, k) not in cells:
                    continue
                entry = _stats(cells[(noise, k)], metric)
                if _interval(entry) is not None:
                    ax.errorbar(k, entry["mean"], yerr=_interval(entry), fmt="none", color=color, capsize=3)
        ax.set(xlabel="Maximum slate size K", ylabel=LABELS[metric], xticks=ks)
        ax.grid(alpha=0.2)
    axes[1].legend(title="Experience noise", fontsize=8)
    fig.suptitle("K-RACER: independently selected policies; marginal 95% mean intervals", fontsize=10)
    return save_figure(fig, output, "current_report/09_grid_K_effects",
                       "Regret versus slate size at each experience-noise level", number=13)


def plot_grid_cutoff_curves(study, output: Path):
    cells, noises, ks = _cells(study)
    fig, axes = plt.subplots(len(noises), len(ks), figsize=(3.25 * len(ks), 2.65 * len(noises)),
                             squeeze=False, constrained_layout=True)
    for row, noise in enumerate(noises):
        for column, k in enumerate(ks):
            ax = axes[row, column]
            case = cells.get((noise, k))
            ax.set_title(f"K={k}, noise={noise:.2f}")
            if case is None:
                ax.text(0.5, 0.5, "N/A: no saved cell", ha="center", transform=ax.transAxes)
                continue
            selected = case["selected_T0"][KEY]
            for metric, prefix, color in zip(METRICS, ("consumed", "slate"), METRIC_COLORS):
                entry = _stats(case,metric)
                ax.errorbar(selected,entry["mean"],yerr=_interval(entry),fmt="o",color=color,
                            capsize=2,label=LABELS[metric])
            ax.text(.5,.98,"Test sweep not saved",ha="center",va="top",fontsize=8,
                    transform=ax.transAxes,color="#6b7480")
            ax.set_xscale("symlog", linthresh=1)
            ax.grid(alpha=0.18)
            if row == len(noises) - 1:
                ax.set_xlabel(r"Post-Startup budget $T_0$")
            if column == 0:
                ax.set_ylabel("Held-out mean regret")
    handles = [plt.Line2D([], [], color=color, label=LABELS[metric])
               for metric, color in zip(METRICS, METRIC_COLORS)]
    handles.append(plt.Line2D([], [], color="#4c5663", marker="o", linestyle="none",
                             label="Validation-selected budget"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3, fontsize=9)
    entry = save_figure(fig, output, "current_report/10_grid_cutoff_curves",
                       "Selected-budget grid costs (test sweeps unavailable)", number=20)
    entry["status"] = "partial: test outcomes exist only at selected budget"
    return entry


def plot_grid_phase_costs(study, output: Path):
    cells, noises, ks = _cells(study)
    all_phases = set()
    data = {}
    for coordinate, case in cells.items():
        for metric in METRICS:
            phases = phase_values(case, KEY, metric)
            data[(coordinate, metric)] = phases
            all_phases.update(phases)
    order = [p for p in PHASE_ORDER if p in all_phases]
    order.extend(sorted(all_phases - set(order)))
    fig, axes = plt.subplots(2, len(noises), figsize=(4.05 * len(noises), 6.2),
                             squeeze=False, constrained_layout=True)
    positions = np.arange(len(ks))
    for row, metric in enumerate(METRICS):
        for column, noise in enumerate(noises):
            ax = axes[row, column]
            bottom = np.zeros(len(ks))
            for phase in order:
                values = []
                for k in ks:
                    if (noise, k) not in cells:
                        values.append(np.nan)
                        continue
                    phase_array = data[((noise, k), metric)].get(phase)
                    values.append(float(np.mean(phase_array)) if phase_array is not None else 0.0)
                values = np.asarray(values)
                ax.bar(positions, values, bottom=bottom, width=0.7,
                       color=PHASE_COLORS.get(phase, "#858d97"),
                       label=PHASE_LABELS.get(phase, phase))
                bottom += np.nan_to_num(values)
            for index, k in enumerate(ks):
                if (noise, k) not in cells:
                    ax.text(index, 0, "N/A", ha="center", va="bottom", color="#56616c")
            ax.set(title=f"Noise {noise:.2f}", xlabel="Maximum slate size K", xticks=positions,
                   xticklabels=ks, ylabel=LABELS[metric])
            ax.grid(axis="y", alpha=0.18)
    handles = [Patch(facecolor=PHASE_COLORS.get(p, "#858d97"), label=PHASE_LABELS.get(p, p)) for p in order]
    fig.legend(handles=handles, loc="outside upper center", ncol=min(len(order), 5), fontsize=9)
    return save_figure(fig, output, "current_report/11_grid_phase_costs",
                       "K-RACER phase regret across the K/noise grid", number=21)


def render_grid(study: dict, output: Path) -> list[dict]:
    """Render only grid figures referenced in the paper's simulation sections."""
    with plt.rc_context(STYLE):
        return [
            plot_grid_selected_parameters(study, output),
            plot_grid_k_effects(study, output),
            plot_grid_cutoff_curves(study, output),
            plot_grid_phase_costs(study, output),
        ]
