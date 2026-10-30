"""Attach verified supplemental observations and refresh only four requested figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from .common import CASE_LABELS,METRICS,mean_ci
from .data import sha256,write_csv,grid_view

RAW_NAMES = dict(zip(METRICS,("consumed_regret","slate_average_regret")))


def cached_study(report):
    """Reconstruct the exact gallery's plotting data from its lossless CSV exports."""
    report = Path(report)
    manifest = json.loads((report/"manifest.json").read_text(encoding="utf-8"))
    hashes = {k.replace("\\","/"):v for k,v in manifest["outputs"].items()}
    for name in ("raw_selected.csv","configurations.csv"):
        if sha256(report/"data"/name) != hashes["data/"+name]:
            raise ValueError("Saved gallery CSV changed: "+name)
    with (report/"data/configurations.csv").open(encoding="utf-8") as stream:
        config_rows = list(csv.DictReader(stream))
    records = {}
    with (report/"data/raw_selected.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            records.setdefault((row["case"],row["method"],row["metric"]),{})[int(row["seed"])] = float(row["value"])
    study = {"cases":{}}
    for row in config_rows:
        cid,key = row["case"],row["method"]
        config = {k:int(row[k]) if k in ("D","K","T") else float(row[k])
                  for k in ("D","K","T","sigma_u","gamma_cara","delta_conf","startup_alpha_deg")}
        case = study["cases"].setdefault(cid,{"id":cid,"config":config,"raw_selected":{},
            "keys":[],"selected_T0":{},"selected_parameters":{}})
        seeds = sorted(records[cid,key,METRICS[0]])
        case["keys"].append(key)
        case["raw_selected"][key] = {"seeds":seeds,"finals":{
            metric:[records[cid,key,metric][s] for s in seeds] for metric in METRICS}}
        case["selected_T0"][key] = int(row["T0"]) if row["T0"] else None
        case["selected_parameters"][key] = {"value":float(row["value"]) if row["value"] and row["value"] != "fixed" else None}
        CASE_LABELS.setdefault(cid,cid)
    return study,manifest


def attach_supplement(study,path,report=None):
    """Verify configuration identity, fixed gamma, held-out seeds, and saved totals."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    budget_records = {}
    for record in payload["records"]:
        job = record["job"]
        if job["kind"] != "budget":
            continue
        case = study["cases"][job["case"]]
        for name,value in job["config"].items():
            np.testing.assert_allclose(case["config"][name],value,rtol=1e-12,atol=1e-12,
                                       err_msg="Supplement belongs to a different simulation configuration")
        if job["gamma"] != case["selected_parameters"]["kracer"]["value"] or job["split"] != "test":
            raise ValueError("Budget sweep changed validation-selected gamma or split")
        key = (job["case"],job["seed"])
        if key in budget_records:
            raise ValueError("Duplicate budget seed")
        budget_records[key] = record
    max_error = 0.
    for cid,case in study["cases"].items():
        relevant = [r for (name,_),r in budget_records.items() if name == cid]
        if not relevant:
            continue
        expected = case["raw_selected"]["kracer"]["seeds"]
        if {r["job"]["seed"] for r in relevant} != set(expected):
            raise ValueError("Incomplete held-out budget seed set: "+cid)
        reference_budgets = relevant[0]["job"]["budgets"]
        if any(r["job"]["budgets"] != reference_budgets for r in relevant):
            raise ValueError("Mismatched saved budget grids")
        curves = {b:{"seeds":[],**{metric:[] for metric in METRICS}} for b in reference_budgets}
        for record in sorted(relevant,key=lambda r:r["job"]["seed"]):
            job = record["job"]
            if [r["T0"] for r in record["result"]] != reference_budgets:
                raise ValueError("Missing budget observations")
            for row in record["result"]:
                if sum(row["phase_counts"].values()) != case["config"]["T"]:
                    raise ValueError("Budget phase counts do not sum to horizon")
                point = curves[row["T0"]]
                point["seeds"].append(job["seed"])
                for metric,raw in RAW_NAMES.items():
                    point[metric].append(row[raw])
        selected = curves[case["selected_T0"]["kracer"]]
        for metric in METRICS:
            raw = case["raw_selected"]["kracer"]
            old = [raw["finals"][metric][raw["seeds"].index(s)] for s in selected["seeds"]]
            max_error = max(max_error,float(np.max(np.abs(np.asarray(old)-selected[metric]))))
            np.testing.assert_allclose(old,selected[metric],rtol=1e-10,atol=2e-5,
                err_msg="Selected-policy replay differs from the saved campaign")
        case["budget_sweep"] = curves
    baseline = study["cases"]["baseline"]
    for fraction in (.25,.75):
        selected = payload["startup_selections"][str(fraction)]
        rows = [r for r in payload["records"] if r["job"]["kind"] == "startup_test" and r["job"]["fraction"] == fraction]
        if {r["job"]["seed"] for r in rows} != set(baseline["raw_selected"]["kracer"]["seeds"]):
            raise ValueError("Incomplete Startup test seeds")
        scores = {}
        for record in payload["records"]:
            job = record["job"]
            if job["kind"] == "startup_validation" and job["fraction"] == fraction:
                for row in record["result"]:
                    scores.setdefault((row["T0"],job["gamma"]),[]).append(row["slate_average_regret"])
        best = min(scores,key=lambda p:(np.mean(scores[p]),p[0],p[1]))
        if best != (selected["T0"],selected["gamma"]):
            raise ValueError("Startup policy not selected from validation data")
        if any(r["job"]["gamma"] != best[1] or r["job"]["budgets"] != [best[0]] for r in rows):
            raise ValueError("Startup test policy differs from validation selection")
        rows.sort(key=lambda r:r["job"]["seed"])
        cfg = dict(baseline["config"],startup_alpha_deg=rows[0]["job"]["startup_angle"])
        cid = f"radius_{fraction}"
        study["cases"][cid] = {"id":cid,"config":cfg,"keys":["kracer"],
            "selected_T0":{"kracer":best[0]},"selected_parameters":{"kracer":{"value":best[1]}},
            "raw_selected":{"kracer":{"seeds":[r["job"]["seed"] for r in rows],
                "finals":{metric:[r["result"][0][raw] for r in rows] for metric,raw in RAW_NAMES.items()}}}}
    historical = []
    for record in payload["records"]:
        job = record["job"]
        if job["kind"] == "historical":
            result = record["result"]
            historical.append({"case":job["case"],"seed":job["seed"],"settings":job,
                "summary":{metric:result[raw] for metric,raw in RAW_NAMES.items()} |
                          {"phase_counts":result["phase_counts"],"precision_stop":result["precision_stop"]}})
    grouped = {}
    for r in historical:
        grouped.setdefault(r["case"],[]).append(r["seed"])
        if sum(r["summary"]["phase_counts"].values()) != r["settings"]["horizon"]:
            raise ValueError("Historical phase counts do not sum to horizon")
        if r["settings"]["model"] == "bayesian" and sum(r["summary"]["phase_counts"].get(p,0) for p in ("excite","localize")) != 64:
            raise ValueError("Historical learning exploration does not sum to T0=64")
    for seeds in grouped.values():
        if len(set(seeds)) != len(seeds) or len(seeds) != len(payload["startup_test_seeds"]):
            raise ValueError("Historical seed counts differ")
    study["historical_records"] = historical
    study["additional"] = payload
    study["selected_replay_max_abs_error"] = max_error
    return study


def curve_stats(case,metric):
    curves = case.get("budget_sweep")
    if not curves:
        raise ValueError("Full held-out budget sweep not saved; run plotting.collect_additional")
    budgets = sorted(curves)
    summaries = [mean_ci(curves[b][metric]) for b in budgets]
    return np.array(budgets),np.array([s["mean"] for s in summaries]),np.array([s["lo"] for s in summaries]),np.array([s["hi"] for s in summaries])


def export_additional(study,output):
    rows = []
    for item in study["additional"]["records"]:
        job,result = item["job"],item["result"]
        for row in result if isinstance(result,list) else [result]:
            cfg = job.get("config",{})
            rows.append({"kind":job["kind"],"case":job["case"],"seed":job["seed"],"split":job["split"],
                "D":cfg.get("D",3),"K":cfg.get("K",job.get("K")),
                "normalized_noise":cfg["sigma_u"]*math.sqrt(cfg["D"]) if cfg else job["noise"],
                "risk":cfg.get("gamma_cara",job.get("risk")),"startup_fraction":job.get("fraction"),
                "T0":row.get("T0",job.get("T0")),"gamma_ctr":job["gamma"],
                "consumed_regret":row["consumed_regret"],"slate_average_regret":row["slate_average_regret"],
                "phase_counts":json.dumps(row["phase_counts"],sort_keys=True)})
    tables = [write_csv(Path(output)/"data/additional_raw_results.csv",rows)]
    summaries = []
    for cid,case in study["cases"].items():
        for budget,values in case.get("budget_sweep",{}).items():
            for metric in METRICS:
                summaries.append({"case":cid,"T0":budget,"metric":metric,
                    "gamma_ctr":case["selected_parameters"]["kracer"]["value"],**mean_ci(values[metric])})
    tables.append(write_csv(Path(output)/"data/full_budget_sweeps.csv",summaries))
    startup = []
    for cid in ("radius_0.25","baseline","radius_0.75"):
        case = study["cases"][cid]
        cfg = case["config"]
        lower = math.degrees(math.acos(1/math.sqrt(cfg["D"])))
        for metric in METRICS:
            startup.append({"case":cid,"fraction":(cfg["startup_alpha_deg"]-lower)/(90-lower),
                "angle_degrees":cfg["startup_alpha_deg"],"T0":case["selected_T0"]["kracer"],
                "gamma_ctr":case["selected_parameters"]["kracer"]["value"],"metric":metric,
                **mean_ci(case["raw_selected"]["kracer"]["finals"][metric])})
    tables.append(write_csv(Path(output)/"data/startup_sensitivity_summary.csv",startup))
    history = []
    for cid in sorted({r["case"] for r in study["historical_records"]}):
        group = [r for r in study["historical_records"] if r["case"] == cid]
        cfg = group[0]["settings"]
        history.append({"case":cid,"K":cfg["K"],"risk":cfg["risk"],"model":cfg["model"],
            **mean_ci([r["summary"][METRICS[0]] for r in group]),
            "mean_excitation_epochs":float(np.mean([r["summary"]["phase_counts"].get("excite",0) for r in group])),
            "mean_localization_epochs":float(np.mean([r["summary"]["phase_counts"].get("localize",0) for r in group])),
            "precision_stops":sum(r["summary"]["precision_stop"] is not None for r in group)})
    tables.append(write_csv(Path(output)/"data/historical_supplied_cone_summary.csv",history))
    return tables


def refresh(report,supplement):
    """Change only 18/20/22/24; assert all other plot assets stay byte-identical."""
    from .main_plots import _sensitivity,_cutoff_tradeoff
    from .grid_plots import plot_grid_cutoff_curves
    from .historical_plots import render_historical
    from .__main__ import _report
    from .common import plt,STYLE
    report = Path(report)
    study,manifest = cached_study(report)
    attach_supplement(study,supplement,report)
    changed = {18,20,22,24}
    previous_articles = {int(n):body for n,body in re.findall(r'<article id="figure-(\d+)">(.*?)</article>',
        (report/"index.html").read_text(encoding="utf-8"),flags=re.S) if int(n) not in changed}
    preserved = {report/path:sha256(report/path) for f in manifest["figures"]
                 if f["paper_figure"] not in changed for path in f["files"].values()}
    with plt.rc_context(STYLE):
        figures = [_sensitivity(study,report),plot_grid_cutoff_curves(grid_view(study),report),
                   _cutoff_tradeoff(study,report),*render_historical(study,report)]
    manifest["figures"] = [f for f in manifest["figures"] if f["paper_figure"] not in changed]+figures
    for item in manifest["coverage"]:
        if item["paper_figure"] in changed:
            item["status"] = "generated"
    tables = export_additional(study,report)
    names = {t["file"] for t in tables}
    manifest["tables"] = [t for t in manifest["tables"] if t["file"] not in names]+tables
    write_csv(report/"data/figure_coverage.csv",manifest["coverage"])
    _report(report,{"root":manifest["input_root"]},manifest["figures"],manifest["coverage"],manifest["tables"])
    new_articles = {int(n):body for n,body in re.findall(r'<article id="figure-(\d+)">(.*?)</article>',
        (report/"index.html").read_text(encoding="utf-8"),flags=re.S) if int(n) not in changed}
    assert previous_articles == new_articles,"An unrequested figure's HTML changed"
    assert all(sha256(path) == digest for path,digest in preserved.items())
    manifest["additional_simulations"] = {"path":str(Path(supplement).relative_to(report)),"sha256":sha256(supplement),
        "selected_replay_max_abs_error":study["selected_replay_max_abs_error"],"other_figure_hashes_unchanged":True,
        "fresh_supplied_cone_protocol":study["additional"]["historical_protocol"]}
    manifest["outputs"] = {str(p.relative_to(report)).replace("\\","/"):sha256(p)
        for p in report.rglob("*") if p.is_file() and p.name != "manifest.json"}
    (report/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("Updated only Figures 18,20,22,24; preserved",len(preserved),"other plot files.")
    print("Maximum selected-policy replay difference:",study["selected_replay_max_abs_error"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,default=Path(__file__).resolve().parent/"results")
    args = parser.parse_args()
    refresh(args.report,args.report/"additional/supplement.json")
