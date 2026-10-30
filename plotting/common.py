"""Shared plotting conventions and statistics computed from saved observations."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

METRICS = ("cumulative_consumed_item_regret", "cumulative_slate_average_regret")
LABELS = dict(zip(METRICS, ("Consumed-item regret", "Slate-average regret")))
METHODS = {
    "kracer": "K-RACER",
    "analytical_certificate": "K-RACER Ablation: Analytical Directional Certificate",
    "minimum_eigenvector": "K-RACER Ablation: Minimum Eigenvector Excitation",
    "full_dimensional": "K-RACER Ablation: Orthonormal Basis Excitation",
    "maximum_normal": "K-RACER Ablation: Maximum Normal Excitation",
    "without_restart": "K-RACER Ablation: Without Restart",
    "raes": "RAES-reworked",
    "dbgd": "DBGD",
    "doubler": "Doubler",
    "sparring": "Sparring",
}
CASE_LABELS = {
    "baseline": "Baseline", "dimension_3": "D=3", "dimension_9": "D=9",
    "speed_0.05": "Noise=0.05", "speed_0.15": "Noise=0.15",
    "K_3": "K=3", "K_5": "K=5", "K_9": "K=9",
}
PHASE_LABELS = {
    "startup": "Startup", "localize": "Localization", "excite": "Excitation",
    "exploit": "Exploitation", "pending": "Pending Startup",
    "dbgd": "DBGD online learning", "doubler": "Doubler online learning",
    "sparring": "Sparring online learning",
}
PHASE_COLORS = dict(zip(PHASE_LABELS, (
    "#b1b8dc", "#3d9690", "#d69b59", "#6487b1", "#d87485",
    "#d69b59", "#b37fa5", "#b35c43",
)))
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
}


def mean_ci(values):
    """Two-sided marginal 95% Student-t interval; undefined for one observation.

    Empty/nonfinite observations are errors, never silently omitted or zeroed.
    These intervals condition on the previously saved validation selection.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty, finite one-dimensional sample")
    mean = float(values.mean())
    if len(values) == 1:
        return {"n": 1, "mean": mean, "lo": None, "hi": None}
    half = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"n": len(values), "mean": mean, "lo": mean - half, "hi": mean + half}


def finals(case, key, metric):
    return np.asarray(case["raw_selected"][key]["finals"][metric], dtype=float)


def distribution_summary(values):
    """Descriptive sample statistics in original regret units, before log display.

    Quartiles use linear interpolation. Tukey whiskers end at the most extreme
    observations within 1.5 IQR of the box; every observation is plotted as a dot.
    """
    values = np.asarray(values,dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Expected a nonempty, finite one-dimensional sample")
    q1,median,q3 = np.percentile(values,[25,50,75],method="linear")
    iqr = float(q3-q1)
    return {"n":len(values),"mean":float(values.mean()),"median":float(median),
        "q1":float(q1),"q3":float(q3),"iqr":iqr,
        "whisker_low":float(values[values >= q1-1.5*iqr].min()),
        "whisker_high":float(values[values <= q3+1.5*iqr].max())}


def endpoint_distribution_rows(case):
    """Return the numerical values shown in Figure 10 for each method/metric."""
    return [{"case":case.get("id","baseline"),"method":key,"metric":metric,
             **distribution_summary(finals(case,key,metric))}
            for key in METHODS if key in case.get("raw_selected",{}) for metric in METRICS]


def paired_samples(case, left, right, metric):
    """Return seed IDs and left-minus-right observations aligned by seed identity."""
    a, b = (case["raw_selected"][key] for key in (left, right))
    for raw in (a, b):
        if len(set(raw["seeds"])) != len(raw["seeds"]):
            raise ValueError("Duplicate paired seed IDs")
    if set(a["seeds"]) != set(b["seeds"]):
        raise ValueError(f"{case.get('id')}: paired seed sets differ")
    for key,raw in ((left,a),(right,b)):
        if finals(case,key,metric).shape != (len(raw["seeds"]),):
            raise ValueError("Paired endpoint/seed count mismatch")
    indices = {seed: i for i, seed in enumerate(b["seeds"])}
    differences = finals(case, left, metric) - finals(case, right, metric)[
        [indices[seed] for seed in a["seeds"]]
    ]
    return list(a["seeds"]),differences


def paired(case, left, right, metric):
    """Compute left-minus-right mean intervals after explicit seed alignment."""
    _,differences = paired_samples(case,left,right,metric)
    return mean_ci(differences)


def paired_distribution_tables(study):
    """Figure 12's descriptive summaries and individual paired test differences."""
    summaries,samples = [],[]
    ablations = ("analytical_certificate","minimum_eigenvector","full_dimensional",
                 "maximum_normal","without_restart")
    for cid,case in study["cases"].items():
        for key in ablations:
            if key not in case["raw_selected"] or "kracer" not in case["raw_selected"]:
                continue
            try:
                seeds,values = paired_samples(case,key,"kracer",METRICS[1])
            except ValueError:
                continue
            base = {"case":cid,"left":key,"right":"kracer","metric":METRICS[1]}
            summaries.append({**base,**distribution_summary(values)})
            samples.extend({**base,"seed":seed,"difference":float(value)}
                           for seed,value in zip(seeds,values))
    return summaries,samples


def phase_values(case, key, metric):
    """Get exact per-seed phase losses, adding Startup components before CIs.

    Prefer inline episode costs. An older study's attached phase values may be
    used only when its seed IDs match. Missing phase data raises an error.
    """
    raw = case["raw_selected"][key]
    summaries = raw["summaries"]
    if all("phase_regret" in summary for summary in summaries):
        names = sorted(set().union(*(s["phase_regret"][metric] for s in summaries)))
        values = {p: np.array([s["phase_regret"][metric].get(p, 0.) for s in summaries])
                  for p in names}
    else:
        supplement = case.get("phase_regret", {}).get(key)
        if supplement is None or supplement["seeds"] != raw["seeds"]:
            raise ValueError(f"{case.get('id')}/{key}: exact phase data unavailable")
        values = {p: np.asarray(v["values"], dtype=float)
                  for p, v in supplement["metrics"][metric].items()}
    startup = [values.pop(p) for p in ("startup_prepare", "startup_sign") if p in values]
    if startup:
        values["startup"] = np.sum(startup, axis=0)
    ordered = {p: values[p] for p in PHASE_LABELS if p in values}
    ordered.update({p: v for p, v in values.items() if p not in ordered})
    np.testing.assert_allclose(np.sum(list(ordered.values()), axis=0), finals(case, key, metric),
                               rtol=1e-10, atol=1e-7)
    return ordered


def save_figure(fig, output, stem, title, number=None):
    """Save editable vector figures plus a preview, closing each figure reliably."""
    base = Path(output) / "figures" / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    try:
        for suffix in ("pdf", "svg", "png"):
            path = base.with_suffix("." + suffix)
            fig.savefig(path, dpi=190, bbox_inches="tight")
            paths[suffix] = str(path.relative_to(output)).replace("\\", "/")
    finally:
        plt.close(fig)
    return {"stem": stem, "title": title, "paper_figure": number, "files": paths}


def error_half(stats):
    """Return zero visual extent for an undefined interval, without exporting it as zero."""
    if stats["lo"] is None:
        warnings.warn("One seed: 95% interval unavailable; marker only", stacklevel=2)
        return 0.
    return stats["mean"] - stats["lo"]
