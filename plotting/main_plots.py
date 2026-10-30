"""Main-study figures from saved observations, without importing simulation code."""

from __future__ import annotations

import math
from inspect import signature
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from .common import (
    CASE_LABELS, LABELS, METHODS, METRICS, STYLE, finals, mean_ci, paired,
    phase_values, save_figure, distribution_summary, paired_samples,
)


COLORS = {
    "kracer": "#007e87", "analytical_certificate": "#b55a31",
    "minimum_eigenvector": "#8460a2", "full_dimensional": "#468c54",
    "maximum_normal": "#a86d24", "without_restart": "#bc5380", "raes": "#3d6da4",
    "dbgd": "#6a7681", "doubler": "#ad7c27", "sparring": "#7770b5",
}
PHASE_NAMES = {
    "startup": "Startup", "localize": "Localization", "excite": "Excitation",
    "exploit": "Exploitation", "pending": "Pending",
}
PHASE_COLORS = {
    "startup": "#b6b8cf", "localize": "#3e9a8d", "excite": "#d09257",
    "exploit": "#5479aa", "pending": "#ce6d83",
}


def _methods(case):
    return [key for key in METHODS if key in case.get("raw_selected", {})]


def _cases(study):
    stored = study.get("cases", {})
    return [cid for cid in CASE_LABELS if cid in stored]


def _label(key, width=28):
    name = METHODS.get(key, key)
    if name.startswith("K-RACER Ablation: "):
        name = name.removeprefix("K-RACER Ablation: ")
        return "K-RACER ablation:\n" + textwrap.fill(name, width)
    return textwrap.fill(name, width)


def _errors(stat):
    if stat["lo"] is None or stat["hi"] is None:
        return None
    return [[max(0., stat["mean"] - stat["lo"])],
            [max(0., stat["hi"] - stat["mean"])]]


def _endpoint_plot(case, output):
    keys = _methods(case)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, max(5.6, len(keys) * .58)),
                             sharey=True, layout="constrained")
    horizontal = {"orientation":"horizontal"} if "orientation" in signature(axes[0].bxp).parameters else {"vert":False}
    for ax, metric in zip(axes, METRICS):
        all_values = []
        for index, key in enumerate(keys):
            values = finals(case,key,metric)
            stat = distribution_summary(values)
            color = COLORS.get(key,"#607080")
            all_values.extend(values)
            box = {"q1":stat["q1"],"med":stat["median"],"q3":stat["q3"],
                   "whislo":stat["whisker_low"],"whishi":stat["whisker_high"]}
            ax.bxp([box],positions=[index],widths=.36,showfliers=False,patch_artist=True,
                   manage_ticks=False,boxprops={"facecolor":color,"edgecolor":color,"alpha":.22},
                   whiskerprops={"color":color,"linewidth":1.2},
                   capprops={"color":color,"linewidth":1.2},
                   medianprops={"color":color,"linewidth":2.3},**horizontal)
            # Stable vertical offsets reveal every test observation without altering regret.
            jitter = np.random.default_rng(10).permutation(np.linspace(-.115,.115,len(values)))
            ax.scatter(values,index+jitter,s=19,color=color,alpha=.68,
                       edgecolors="white",linewidths=.35,zorder=3)
            ax.scatter(stat["mean"],index,marker="D",s=40,facecolors="white",
                       edgecolors=color,linewidths=1.5,zorder=5)
            description = (f"Mean {stat['mean']:.2f}   Median {stat['median']:.2f}   "
                           f"Q1 {stat['q1']:.2f}   Q3 {stat['q3']:.2f}   IQR {stat['iqr']:.2f}")
            ax.text(.01,index+.34,description,transform=ax.get_yaxis_transform(),
                    fontsize=6.7,color="#515a64",ha="left",va="center")
        ax.set_xscale("log" if np.all(np.asarray(all_values)>0) else "symlog")
        ax.set(xlabel=LABELS[metric], yticks=range(len(keys)))
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticklabels([_label(key) for key in keys])
    axes[0].set_ylim(len(keys)-.35,-.45)
    legend = [Patch(facecolor="#87939f",alpha=.22,label="Q1-Q3 box (IQR)"),
        Line2D([],[],color="#63717f",linewidth=2.3,label="Median line"),
        Line2D([],[],marker="D",markerfacecolor="white",markeredgecolor="#63717f",
               linestyle="none",label="Mean"),
        Line2D([],[],marker="o",color="#63717f",linestyle="none",markersize=4,
               label="Test-seed regret")]
    fig.legend(handles=legend,loc="outside lower center",ncol=4,frameon=False,fontsize=8,
               title="Box width = IQR = Q3 - Q1; whiskers within 1.5 IQR. All test samples shown.",
               title_fontsize=8)
    fig.suptitle("Baseline scenario: final regret distributions")
    entry = save_figure(fig,output,"current_reference","Baseline final regret distributions",number=10)
    entry["description"] = "Boxes show Q1-Q3, the line is the median, diamonds are means, and dots are individual test-seed regrets. Whiskers use 1.5 IQR; numerical summaries are shown beneath each box."
    return entry


def _trajectory_plot(case, output):
    available = case.get("trajectories", {})
    order = ["kracer", "raes", "dbgd", "doubler", "sparring"]
    keys = [key for key in order if key in available]
    if not keys:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), layout="constrained")
    for ax, metric in zip(axes, METRICS):
        for index, key in enumerate(keys):
            saved = available[key]
            series = saved[metric]
            if isinstance(series, dict):
                series = series["mean"]
            rounds, values = np.asarray(saved["rounds"]), np.asarray(series, dtype=float)
            mask = (rounds > 0) & (values > 0) & np.isfinite(values)
            ax.plot(rounds[mask], values[mask], label=METHODS.get(key, key),
                    color=COLORS.get(key), linewidth=2 if key == "kracer" else 1.3,
                    linestyle=["-", "--", ":", "-."][index % 4])
        ax.set(xscale="log", yscale="log", xlabel="Interaction epoch", ylabel=LABELS[metric])
        ax.grid(alpha=.2)
    # A shared legend outside the axes preserves the short-horizon trajectory region.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               fontsize=8, frameon=False)
    fig.suptitle("Baseline scenario: saved mean cumulative regret trajectories")
    return save_figure(fig, output, "current_reference_trajectories", "Baseline mean regret trajectories", number=11)


def _all_method_heatmap(study, output, metric):
    cids = _cases(study)
    keys = [key for key in METHODS if any(key in study["cases"][cid].get("raw_selected", {}) for cid in cids)]
    values = np.full((len(keys), len(cids)), np.nan)
    for col, cid in enumerate(cids):
        case = study["cases"][cid]
        for row, key in enumerate(keys):
            if key in case.get("raw_selected", {}):
                values[row, col] = mean_ci(finals(case, key, metric))["mean"]
    best = np.zeros(values.shape,dtype=bool)
    for col in range(len(cids)):
        available = values[:,col][np.isfinite(values[:,col])]
        if available.size:
            # Rank full-precision test means, rather than their rounded labels.
            best[:,col] = np.isfinite(values[:,col]) & (values[:,col] == available.min())
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size:
        low, high = float(positive.min()), float(positive.max())
        norm = LogNorm(low, high if high > low else low * 1.01)
    else:
        # A perfect zero-regret fixture is still observed data, rather than N/A.
        norm = Normalize(0, 1)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(14.5, max(6.6, len(keys) * .6)), layout="constrained")
    masked = np.ma.masked_where(~np.isfinite(values) | (values <= 0), values)
    mesh = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set(xticks=range(len(cids)), yticks=range(len(keys)))
    ax.set_xticklabels([CASE_LABELS[cid] for cid in cids], rotation=40, ha="right")
    ax.set_yticklabels([_label(key, 30) for key in keys])
    for row in range(len(keys)):
        for col in range(len(cids)):
            value = values[row, col]
            if not np.isfinite(value):
                text, color = "N/A", "#6b7078"
            elif value <= 0:
                text, color = f"{value:,.1f}", "#30343a"
            else:
                text = f"{value:,.1f}"
                color = "white" if mesh.norm(value) > .6 else "#182c36"
            if best[row,col]:
                ax.add_patch(Rectangle((col-.46,row-.44),.92,.88,fill=False,
                                       edgecolor="#b87400",linewidth=2,zorder=4))
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=8.4,
                    fontweight="bold" if best[row,col] else "normal")
    ax.set_title(f"All methods: {LABELS[metric].lower()} (test means)\n"
                 "Outlined bold cells: lowest mean in each scenario", pad=13)
    color_label = "Mean final regret (logarithmic color scale)" if positive.size else "Mean final regret (all observed means are zero)"
    fig.colorbar(mesh, ax=ax, shrink=.78, pad=.025, label=color_label)
    consumed = metric == METRICS[0]
    stem = "current_report/01_all_methods_" + ("consumed" if consumed else "slate")
    entry = save_figure(fig, output, stem, f"All-method {LABELS[metric].lower()}",
                        number=16 if consumed else 17)
    entry["description"] = "Gold outlines and bold labels identify the lowest unrounded test-mean regret in each scenario column. All exact ties are highlighted. N/A cells are excluded from the comparison."
    return entry


def _sensitivity(study, output):
    cases = study["cases"]
    def entries(prefix, field):
        result = []
        for cid,case in cases.items():
            if cid != "baseline" and not cid.startswith(prefix):
                continue
            cfg = case["config"]
            if field == "noise":
                value = cfg["sigma_u"] * np.sqrt(cfg["D"])
            elif field == "startup":
                lower = np.degrees(np.arccos(1/np.sqrt(cfg["D"])))
                value = (cfg["startup_alpha_deg"]-lower)/(90-lower)
            else:
                value = cfg[field]
            result.append((cid,value))
        return sorted(result,key=lambda pair:pair[1])
    families = [
        ("Ambient dimension D", entries("dimension_", "D")),
        ("Normalized experience noise", entries("speed_", "noise")),
        ("Maximum slate size K", entries("K_", "K")),
        ("Startup target fraction", entries("radius_", "startup")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.1), layout="constrained")
    metric_colors = (COLORS["kracer"],"#b85d22")
    metric_markers = ("o","s")
    for col, (xlabel, entries) in enumerate(families):
        entries = [(cid, x) for cid, x in entries if cid in study["cases"] and "kracer" in study["cases"][cid]["raw_selected"]]
        ax = axes[col]
        if len(entries) < 2:
            ax.text(.5,.5,"No sweep saved",ha="center",transform=ax.transAxes,color="#687482")
        for metric,color,marker in zip(METRICS,metric_colors,metric_markers):
            stats = [mean_ci(finals(study["cases"][cid], "kracer", metric)) for cid, _ in entries]
            xs = [x for _, x in entries]
            ax.plot(xs, [s["mean"] for s in stats], color=color, marker=marker, lw=1.4,
                    label=LABELS[metric])
            for x, stat in zip(xs, stats):
                ax.errorbar(x, stat["mean"], yerr=_errors(stat), color=color, capsize=3)
        ax.set(xlabel=xlabel, ylabel="Mean final cumulative regret", xticks=xs)
        ax.grid(alpha=.2)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    fig.suptitle("K-RACER sensitivity: selected policies and marginal 95% intervals")
    entry = save_figure(fig, output, "current_report/02_kracer_sensitivity",
                       "K-RACER scenario sensitivity", number=18)
    entry["description"] = "Four panels compare dimension, experience noise, maximum slate size, and Startup target fraction. Teal circles show consumed-item regret and orange squares show slate-average regret on the same axes. Bars are marginal 95% Student-t mean intervals. Each Startup target selects gamma and the post-Startup budget on validation seeds before held-out evaluation."
    if len(families[3][1]) < 3:
        entry["status"] = "partial: Startup target sweep required"
    return entry


def _paired_ablations(study, output):
    cids = _cases(study)
    ablations = ["analytical_certificate", "minimum_eigenvector", "full_dimensional", "maximum_normal", "without_restart"]
    keys = [key for key in ablations if any(key in study["cases"][cid].get("raw_selected", {}) for cid in cids)]
    cols = 3 if len(keys) > 4 else 2
    if not keys:
        return None
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15.5 if cols == 3 else 12.5, rows * 4.7), layout="constrained", squeeze=False)
    for ax, key in zip(axes.flat, keys):
        ax.axvline(0, color="#444b54", lw=1, linestyle="--")
        for index, cid in enumerate(cids):
            case = study["cases"][cid]
            if key not in case.get("raw_selected", {}) or "kracer" not in case.get("raw_selected", {}):
                ax.text(.98, index, "N/A", transform=ax.get_yaxis_transform(), ha="right", color="#777777")
                continue
            try:
                _,values = paired_samples(case, key, "kracer", METRICS[1])
            except ValueError:
                ax.text(.98,index,"Unpaired seeds",transform=ax.get_yaxis_transform(),ha="right")
                continue
            stat = distribution_summary(values)
            color = COLORS.get(key,"#52728b")
            horizontal = {"orientation":"horizontal"} if "orientation" in signature(ax.bxp).parameters else {"vert":False}
            box = {"q1":stat["q1"],"med":stat["median"],"q3":stat["q3"],
                   "whislo":stat["whisker_low"],"whishi":stat["whisker_high"]}
            ax.bxp([box],positions=[index],widths=.36,showfliers=False,patch_artist=True,
                   manage_ticks=False,boxprops={"facecolor":color,"edgecolor":color,"alpha":.22},
                   whiskerprops={"color":color,"linewidth":1.2},capprops={"color":color,"linewidth":1.2},
                   medianprops={"color":color,"linewidth":2.3},**horizontal)
            jitter = np.random.default_rng(10).permutation(np.linspace(-.115,.115,len(values)))
            ax.scatter(values,index+jitter,s=19,color=color,alpha=.68,
                       edgecolors="white",linewidths=.35,zorder=3)
            ax.scatter(stat["mean"],index,marker="D",s=40,facecolors="white",
                       edgecolors=color,linewidths=1.5,zorder=5)
            description = (f"Mean {stat['mean']:.2f}   Median {stat['median']:.2f}   "
                           f"Q1 {stat['q1']:.2f}   Q3 {stat['q3']:.2f}   IQR {stat['iqr']:.2f}")
            ax.text(.01,index+.34,description,transform=ax.get_yaxis_transform(),
                    fontsize=6.1,color="#515a64",ha="left",va="center",
                    bbox={"facecolor":"white","edgecolor":"none","pad":.3,"alpha":.9})
        ax.set(yticks=range(len(cids)), xlabel="Rival minus K-RACER slate-average regret")
        ax.set_yticklabels([CASE_LABELS[cid] for cid in cids], fontsize=8.5)
        ax.set_ylim(len(cids)-.35,-.45)
        name = METHODS[key].removeprefix("K-RACER Ablation: ")
        ax.set_title(textwrap.fill(name, 32), fontsize=11)
        ax.grid(axis="x", alpha=.2)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    legend = [Patch(facecolor="#87939f",alpha=.22,label="Q1-Q3 box (IQR)"),
        Line2D([],[],color="#63717f",linewidth=2.3,label="Median line"),
        Line2D([],[],marker="D",markerfacecolor="white",markeredgecolor="#63717f",linestyle="none",label="Mean"),
        Line2D([],[],marker="o",color="#63717f",linestyle="none",markersize=4,label="Paired test-seed difference")]
    fig.legend(handles=legend,loc="outside lower center",ncol=4,frameon=False,fontsize=8,
               title="Box width = IQR = Q3 - Q1; whiskers within 1.5 IQR. All paired test samples shown.",title_fontsize=8)
    fig.suptitle("Paired final slate-regret distributions (positive favors K-RACER)")
    entry = save_figure(fig, output, "current_report/03_paired_ablations",
                       "Paired K-RACER ablation regret distributions", number=12)
    entry["description"] = "Each dot is one seed-aligned ablation-minus-K-RACER slate-regret difference. Boxes show Q1-Q3, median lines, mean diamonds, and 1.5 IQR whiskers; numerical summaries appear beneath each box. Positive values favor K-RACER."
    return entry


def _cutoff_tradeoff(study, output):
    from .additional import curve_stats
    cases = [(cid,case) for cid,case in study["cases"].items()
             if (cid == "baseline" or cid.startswith("speed_")) and "kracer" in case["raw_selected"]]
    if not cases:
        return None
    fig,axes = plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    colors = ["#2d8f76","#3d6da4","#b55a31"]
    for i,(cid,case) in enumerate(sorted(cases,key=lambda x:x[1]["config"]["sigma_u"])):
        noise = case["config"]["sigma_u"]*np.sqrt(case["config"]["D"])
        for ax,metric in zip(axes,METRICS):
            if not case.get("budget_sweep"):
                ax.text(.5,.5,"Held-out sweeps not saved",ha="center",transform=ax.transAxes)
                continue
            budgets,means,lo,hi = curve_stats(case,metric)
            color = colors[i%len(colors)]
            ax.plot(budgets,means,color=color,lw=1.6,label=f"Noise {noise:.2f}")
            if all(value is not None for value in lo):
                ax.fill_between(budgets,np.asarray(lo,dtype=float),np.asarray(hi,dtype=float),color=color,alpha=.12)
            selected = case["selected_T0"]["kracer"]
            index = int(np.flatnonzero(budgets == selected)[0])
            ax.plot(selected,means[index],marker="o",markerfacecolor="white",markeredgecolor=color,
                    markeredgewidth=1.5,ms=7,zorder=4)
            ax.set(xlabel="Post-Startup exploration budget $T_0$",ylabel=LABELS[metric])
            ax.set_xscale("symlog",base=2,linthresh=1)
            ax.grid(alpha=.2)
    axes[1].legend(frameon=False)
    fig.suptitle("K-RACER exploration-budget tradeoff: open circles mark validation selections")
    entry = save_figure(fig,output,"current_report/05_cutoff_tradeoff",
                        "Held-out exploration-budget tradeoff across experience-noise levels",number=22)
    entry["description"] = "D=5 and K=2 at the current noise levels. Within each setting, contraction is fixed at its saved validation selection. Curves are test means, shading is a marginal 95% Student-t interval, and circles mark validation-selected budgets. T0 excludes Startup; no test-based selection is performed."
    if any(not c.get("budget_sweep") for _,c in cases):
        entry["status"] = "partial: additional held-out sweeps required"
    return entry


def _main_phases(study, output):
    cids = [cid for cid in _cases(study) if "kracer" in study["cases"][cid].get("raw_selected", {})]
    values = {metric: [phase_values(study["cases"][cid], "kracer", metric) for cid in cids] for metric in METRICS}
    known = list(PHASE_NAMES)
    all_phases = set().union(*(set(v) for entries in values.values() for v in entries))
    phases = [p for p in known if p in all_phases] + sorted(all_phases - set(known))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3), sharey=True, layout="constrained")
    for ax, metric in zip(axes, METRICS):
        left = np.zeros(len(cids))
        for phase in phases:
            means = np.asarray([np.mean(v[phase]) if phase in v else 0. for v in values[metric]])
            ax.barh(range(len(cids)), means, left=left, label=PHASE_NAMES.get(phase, phase),
                    color=PHASE_COLORS.get(phase, "#a8acb4"), height=.7)
            left += means
        expected = [mean_ci(finals(study["cases"][cid], "kracer", metric))["mean"] for cid in cids]
        np.testing.assert_allclose(left, expected, rtol=1e-10, atol=1e-7,
                                   err_msg="Stacked phase costs must equal final regret")
        ax.set(xlabel=LABELS[metric], yticks=range(len(cids)))
        ax.grid(axis="x", alpha=.18)
    axes[0].set_yticklabels([CASE_LABELS[cid] for cid in cids])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(5, len(phases)), frameon=False)
    fig.suptitle("K-RACER phase regret: startup, localization, excitation, and exploitation")
    return save_figure(fig, output, "current_report/06_main_phase_costs",
                       "K-RACER phase costs across scenarios", number=23)


def render_main(study: dict, output: Path) -> list[dict]:
    """Render every main-study empirical figure using only the supplied study.

    Endpoints and paired intervals are recomputed from held-out seeds.
    Budget points retain each policy's saved validation-selected settings.
    The caller controls the output root; no input or simulation file is written.
    """
    figures = []
    with plt.rc_context(STYLE):
        reference = study.get("cases", {}).get("baseline")
        if reference:
            figures.append(_endpoint_plot(reference, output))
            figures.append(_trajectory_plot(reference, output))
        for metric in METRICS:
            figures.append(_all_method_heatmap(study, output, metric))
        figures.append(_sensitivity(study, output))
        figures.append(_paired_ablations(study, output))
        figures.append(_cutoff_tradeoff(study, output))
        figures.append(_main_phases(study, output))
    return [item for item in figures if item is not None]
