"""Read native seed_*.json / validation.json files without importing src.

Each file is consumed once and discarded after extracting regret arrays. This
keeps the multi-gigabyte raw slates out of memory. Nothing is written to inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .common import CASE_LABELS, METHODS, METRICS, finals, mean_ci, paired, phase_values

RAW_METRICS = dict(zip(METRICS, ("consumed_regret", "slate_average_regret")))


class MissingDataError(ValueError):
    """A requested plot requires observations that were not saved."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path, inputs):
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input is still being written: {path}")
    inputs[str(path.resolve())] = {"sha256": hashlib.sha256(content).hexdigest(),
        "size": after.st_size, "mtime_ns": after.st_mtime_ns}
    return json.loads(content)


def _validation(payload):
    if "error" in payload:
        raise ValueError("Validation failed: " + payload["error"])
    seeds = payload["validation_seeds"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate validation seed")
    if set(seeds) & set(payload.get("test_seeds", [])):
        raise ValueError("Validation/test seed overlap")
    for score in payload["scores"]:
        if len(score["seed_scores"]) != len(seeds):
            raise ValueError("Validation score/seed count mismatch")
        np.testing.assert_allclose(mean_ci(score["seed_scores"])["mean"],
                                   score["mean_slate_regret"], rtol=1e-10, atol=1e-7)
    best = min(payload["scores"], key=lambda s: (s["mean_slate_regret"], s["cutoff"], s["parameter"]))
    if (best["cutoff"], best["parameter"]) != (payload["selected_cutoff"], payload["selected_parameter"]):
        raise ValueError("Saved selection differs from validation minimum")


def _episode(payload):
    """Validate the actual manual format and extract per-round / per-phase losses."""
    if "error" in payload:
        raise ValueError(payload["error"])
    settings, summary, rows = payload["settings"], payload["summary"], payload["rounds"]
    if len(rows) != settings["horizon"]:
        raise ValueError("Incomplete episode: rounds do not equal horizon")
    times = np.asarray([r["t"] for r in rows], dtype=int)
    if not np.array_equal(times, np.arange(1, len(rows) + 1)):
        raise ValueError("Round numbers must be exactly 1 through horizon")
    phases = [r["phase"].lower() for r in rows]
    counts = dict(Counter(phases))
    if counts != {p.lower(): n for p, n in summary["phase_counts"].items()}:
        raise ValueError("Saved phase counts differ from recorded rounds")
    result = {"seed": payload["seed"], "phase_counts": counts, "phase_regret": {},
              "startup_rounds": payload["startup"]["rounds"],
              "startup_certified": payload["startup"]["status"] == "CERTIFIED",
              "absolute_cutoff": settings.get("absolute_cutoff")}
    costs = {}
    for metric, raw_name in RAW_METRICS.items():
        values = np.asarray([r[raw_name] for r in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite round regret")
        total = float(values.sum())
        np.testing.assert_allclose(total, summary[raw_name], rtol=1e-10, atol=1e-7)
        result[metric] = total
        result["phase_regret"][metric] = {
            p: float(values[np.array(phases) == p].sum()) for p in counts}
        costs[metric] = np.cumsum(values)
    return result, costs


def load_run(root, *, include_incomplete=False, trajectory_stride=10, progress=None):
    """Load one serial or parallel manual campaign; reject ambiguous merged runs.

    Successful, complete method/case groups are used by default. Failed/missing
    episodes are recorded in ``issues``; --include-incomplete opts into partial
    groups. Never fill missing observations or test budget curves with zeros.
    """
    root = Path(root).resolve()
    if trajectory_stride < 1:
        raise ValueError("trajectory_stride must be positive")
    groups = defaultdict(list)
    for path in root.rglob("seed_*.json"):
        groups[path.parent].append(path)
    if not groups:
        raise MissingDataError(f"No seed_*.json files under {root}")
    study = {"cases": {}, "inputs": {}, "issues": [], "root": str(root)}
    for number, (directory, paths) in enumerate(sorted(groups.items()), 1):
        if progress:
            progress(f"Reading group {number}/{len(groups)}: {directory.relative_to(root)}")
        validation_path = directory / "validation.json"
        validation = None
        if validation_path.exists():
            try:
                validation = _read(validation_path, study["inputs"])
                _validation(validation)
            except (ValueError, AssertionError, KeyError) as error:
                study["issues"].append({"path": str(validation_path), "status": "invalid validation", "detail": str(error)})
                continue
        records, curves, config, case_id, key, label = [], defaultdict(list), None, None, None, None
        truth = {}
        for path in sorted(paths):
            try:
                payload = _read(path, study["inputs"])
                summary, costs = _episode(payload)
                settings = {k:v for k,v in payload["settings"].items() if k != "absolute_cutoff"}
                if case_id is None:
                    case_id, key, label, config = payload["case"], payload["algorithm"], payload.get("label"), settings
                if (case_id, key, config) != (payload["case"], payload["algorithm"], settings):
                    raise ValueError("Mixed cases, algorithms or settings in one group")
                if validation:
                    parameter_name = validation["parameter_name"]
                    if settings["post_startup_cutoff"] != validation["selected_cutoff"] or settings[parameter_name] != validation["selected_parameter"]:
                        raise ValueError("Episode does not use saved validation-selected settings")
                    if summary["seed"] not in validation["test_seeds"]:
                        raise ValueError("Unexpected test seed")
                if summary["seed"] in truth:
                    raise ValueError("Duplicate seed ID in group")
                truth[summary["seed"]] = payload["preference"]
                records.append(summary)
                for metric, values in costs.items():
                    curves[metric].append(values)
            except (ValueError, AssertionError, KeyError) as error:
                study["issues"].append({"path": str(path), "status": "invalid episode", "detail": str(error)})
            finally:
                if "payload" in locals():
                    del payload
        expected = validation["test_seeds"] if validation else [s["seed"] for s in records]
        actual = [s["seed"] for s in records]
        complete = bool(records) and set(actual) == set(expected)
        if not complete:
            study["issues"].append({"path": str(directory), "status": "incomplete group",
                "detail": f"{len(actual)} successful / {len(expected)} expected seeds; missing {sorted(set(expected)-set(actual))}"})
            if not include_incomplete or not records:
                continue
        if not records:
            continue
        canonical = {"D": config["dimension"], "K": config["slate_size"], "T": config["horizon"],
            "sigma_u": config["sigma_u"], "gamma_cara": config["risk"], "delta_conf": config["delta_conf"],
            "startup_alpha_deg": float(np.degrees(config["startup_half_angle_radians"])),
            "user_model": "bayesian", "startup_regime": "sphere"}
        case = study["cases"].setdefault(case_id, {"id": case_id, "label": label or case_id,
            "config": canonical, "keys": [], "raw_selected": {}, "statistics": {}, "trajectories": {},
            "selected_T0": {}, "selected_parameters": {}, "validation": {}, "complete": {},
            "validation_seeds": [], "test_seeds": [], "source_directories": {}})
        if canonical != case["config"]:
            raise ValueError(f"Case {case_id} has inconsistent settings; pass one campaign directory")
        if key in case["raw_selected"]:
            raise ValueError(f"Duplicate {case_id}/{key}; select one campaign with --input")
        for other in case["raw_selected"].values():
            for seed in set(actual) & set(other["seeds"]):
                np.testing.assert_allclose(truth[seed], other["preferences"][seed], atol=1e-12)
        case["keys"].append(key)
        case["raw_selected"][key] = {"seeds": actual, "summaries": records, "preferences": truth,
            "finals": {m: [s[m] for s in records] for m in METRICS}}
        case["statistics"][key] = {m: mean_ci(case["raw_selected"][key]["finals"][m]) for m in METRICS}
        times = np.unique(np.r_[np.arange(1, min(128, canonical["T"]) + 1),
                               np.arange(1, canonical["T"] + 1, trajectory_stride), canonical["T"]])
        trajectory = {"rounds": times.tolist()}
        for metric in METRICS:
            matrix = np.asarray(curves[metric])[:, times - 1]
            means = matrix.mean(axis=0)
            if len(records) > 1:
                from scipy.stats import t
                half = t.ppf(.975, len(records)-1) * matrix.std(axis=0, ddof=1) / np.sqrt(len(records))
                trajectory[metric] = {"mean": means.tolist(), "lo": (means-half).tolist(), "hi": (means+half).tolist()}
            else:
                trajectory[metric] = {"mean": means.tolist(), "lo": [None]*len(times), "hi": [None]*len(times)}
        case["trajectories"][key] = trajectory
        case["selected_T0"][key] = config["post_startup_cutoff"]
        parameter_name = validation["parameter_name"] if validation else None
        case["selected_parameters"][key] = {"parameter": parameter_name,
            "value": validation["selected_parameter"] if validation else None}
        case["validation"][key] = validation
        case["complete"][key] = complete
        case["source_directories"][key] = str(directory)
        case["validation_seeds"] = sorted(set(case["validation_seeds"]) | set(validation["validation_seeds"] if validation else []))
        case["test_seeds"] = sorted(set(case["test_seeds"]) | set(actual))
        if key not in METHODS:
            METHODS[key] = key.replace("_", " ")
        CASE_LABELS.setdefault(case_id, label or case_id)
    if not study["cases"]:
        raise MissingDataError("No usable complete episodes; inspect validation/error JSON, or use --include-incomplete")
    return study


def grid_view(study):
    """Use only D=5, same-risk K/noise cases, preserving absent grid cells."""
    baseline = study["cases"].get("baseline")
    cfg = baseline["config"] if baseline else next(iter(study["cases"].values()))["config"]
    cases = {cid: case for cid, case in study["cases"].items()
             if case["config"]["D"] == cfg["D"]
             and case["config"]["gamma_cara"] == cfg["gamma_cara"]
             and case["config"]["startup_alpha_deg"] == cfg["startup_alpha_deg"]
             and case["config"]["T"] == cfg["T"] and "kracer" in case["raw_selected"]}
    return {"cases": cases}


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = iter(rows)
    first = next(rows, None)
    if fields is None:
        fields = list(first) if first is not None else ["status"]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in itertools.chain([first] if first is not None else [], rows):
            writer.writerow(row)
            count += 1
    return {"file": path.name, "rows": count}


def export_csvs(study, output):
    output = Path(output) / "data"
    selected, raw, phases, pairs, pair_status, validations, configurations = [], [], [], [], [], [], []
    for cid, case in study["cases"].items():
        for key in case["keys"]:
            observations = case["raw_selected"][key]
            base = {"case": cid, "method": key, **case["config"],
                "normalized_noise": case["config"]["sigma_u"] * np.sqrt(case["config"]["D"]),
                "T0": case["selected_T0"][key], "complete": case["complete"][key],
                **case["selected_parameters"][key]}
            configurations.append({**base, "source_directory": case["source_directories"][key]})
            for metric in METRICS:
                selected.append({**base, "metric": metric, **mean_ci(finals(case, key, metric))})
                for seed, value in zip(observations["seeds"], finals(case, key, metric)):
                    raw.append({"case": cid, "method": key, "seed": seed, "metric": metric, "value": value})
                for phase, values in phase_values(case, key, metric).items():
                    counts = [s["phase_counts"].get(phase, 0) for s in observations["summaries"]]
                    phases.append({"case": cid, "method": key, "metric": metric, "phase": phase,
                        **mean_ci(values), **{"count_"+k: v for k,v in mean_ci(counts).items()}})
            validation = case["validation"][key]
            if validation:
                for score in validation["scores"]:
                    validations.append({"case": cid, "method": key, "parameter_name": validation["parameter_name"],
                        "parameter_value": score["parameter"], "T0": score["cutoff"],
                        "selected": score["cutoff"] == validation["selected_cutoff"] and score["parameter"] == validation["selected_parameter"],
                        "metric": METRICS[1], "split": "validation", **mean_ci(score["seed_scores"])})
        for a,b in itertools.combinations(case["keys"],2):
            left, right = (b,a) if a == "kracer" else (a,b)
            for metric in METRICS:
                try:
                    stats = paired(case,left,right,metric)
                except ValueError as error:
                    pair_status.append({"case":cid,"left":left,"right":right,"status":str(error)})
                    continue
                pairs.append({"case":cid,"left":left,"right":right,"metric":metric,**stats})
    tables = [write_csv(output/(name+".csv"), rows) for name,rows in (
        ("selected_results", selected),("raw_selected", raw),("phase_costs",phases),
        ("paired_differences",pairs),("pairing_status",pair_status),("validation_scores",validations),
        ("configurations",configurations),("input_issues",study["issues"]))]
    def trajectories():
        for cid,case in study["cases"].items():
            for key,series in case["trajectories"].items():
                for metric in METRICS:
                    for i,epoch in enumerate(series["rounds"]):
                        yield {"case":cid,"method":key,"metric":metric,"epoch":epoch,
                            **{field:series[metric][field][i] for field in ("mean","lo","hi")}}
    tables.append(write_csv(output/"trajectories.csv",trajectories()))
    return tables
