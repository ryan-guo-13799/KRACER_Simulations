"""Plot the paper's simulation figures from native manual simulation JSON."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import platform

from .data import MissingDataError, export_csvs, grid_view, load_run, sha256, write_csv

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIGURES = {
    10:"Baseline endpoint regrets", 11:"Baseline regret trajectories", 12:"Paired ablations",
    13:"Regret versus K and noise", 16:"All-method consumed regret", 17:"All-method slate regret",
    18:"K-RACER sensitivity", 19:"Selected grid parameters", 20:"Test budget curves across the grid",
    21:"Grid phase costs", 22:"Test exploration-budget tradeoff", 23:"Main phase costs",
    24:"Historical supplied-cone simulations",
}


def default_input():
    output = PROJECT/"output"
    runs = [p for p in output.iterdir() if p.is_dir() and next(p.rglob("seed_*.json"),None)] if output.is_dir() else []
    if not runs:
        raise MissingDataError(f"No saved seed files in {output}; pass --input RUN_DIRECTORY")
    return max(runs,key=lambda p:p.stat().st_mtime_ns)


def _source_hashes():
    return {str(p.relative_to(PROJECT)):sha256(p) for p in (PROJECT/"src").rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def _report(output, study, figures, coverage, tables):
    main_blocks, appendix_blocks = [], []
    for f in sorted(figures,key=lambda x:x["paper_figure"]):
        number = f["paper_figure"]
        if number not in FIGURES:
            continue
        links = " | ".join(f'<a href="{p}">{fmt.upper()}</a>' for fmt,p in f["files"].items())
        title = f"Figure {f['paper_figure']}: {f['title']}"
        block = f'<article id="figure-{number}"><h2>{html.escape(title)}</h2><p>{links}</p><img src="{f["files"]["png"]}" alt="{html.escape(title)}"></article>'
        (main_blocks if number in (10,11,12,13) else appendix_blocks).append(block)
    csv_links = "".join(f'<li><a href="data/{t["file"]}">{t["file"]}</a> ({t["rows"]:,} rows)</li>' for t in tables)
    unavailable = "".join(f'<li>Figure {r["paper_figure"]}: {html.escape(r["status"])}</li>' for r in coverage if r["status"] != "generated")
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Manual simulation results</title>
<style>body{{font:16px/1.5 system-ui;max-width:1250px;margin:auto;padding:30px;background:#eef2f5;color:#203044}}article,section{{background:white;padding:22px;margin:24px 0;border-radius:8px}}img{{max-width:100%}}h2{{font-size:20px}}a{{color:#076f87}}</style>
<h1>Manual simulation figures</h1>
<nav><a href="#main-figures">Main: Figures 10-13</a> | <a href="#appendix-figures">Simulation appendix</a> | <a href="#csv-results">CSV results</a></nav>
<h2 id="main-figures">Main simulation: Figures 10-13</h2>
{''.join(main_blocks)}
<h2 id="appendix-figures">Simulation appendix</h2>
{''.join(appendix_blocks)}
<section><h2>Saved-data coverage</h2><ul>{unavailable}</ul><p>Absent method/case combinations are N/A. Noise values and K axes come from the manual settings. Incomplete or failed groups are recorded in input_issues.csv; partial groups are excluded by default.</p></section>
<section id="csv-results"><h2>CSV results</h2><ul>{csv_links}</ul><a href="manifest.json">Input hashes and provenance</a>
<p>Input: {html.escape(study['root'])}</p><p>Values come exclusively from the selected manual run's saved seed and validation files. Intervals are marginal 95% Student-t intervals.</p></section></html>'''
    (output/"index.html").write_text(text,encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,help="One saved manual campaign directory; defaults to newest output run")
    parser.add_argument("--output",type=Path,default=HERE/"results",help="Destination inside plotting/ (default plotting/results)")
    parser.add_argument("--include-incomplete",action="store_true",help="Include successful seeds from incomplete groups, with their actual n")
    parser.add_argument("--trajectory-stride",type=int,default=10,help="Trajectory sampling stride; first 128 epochs and endpoint always retained (default 10; use 1 for all)")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output == HERE or output.is_relative_to(HERE/"tests"):
        parser.error("Output must be a separate generated-results directory inside plotting/")
    try:
        source = (args.input or default_input()).resolve()
        if source.is_relative_to(output) or output.is_relative_to(source):
            parser.error("Input and output directories must be separate")
        before = _source_hashes()
        study = load_run(source,include_incomplete=args.include_incomplete,
                         trajectory_stride=args.trajectory_stride,progress=lambda s:print(s,flush=True))
    except (ValueError,OSError) as error:
        parser.error(str(error))
    output.mkdir(parents=True,exist_ok=True)
    tables = export_csvs(study,output)
    from .main_plots import render_main
    from .grid_plots import render_grid
    from .historical_plots import render_historical
    figures = render_main(study,output)
    grid = grid_view(study)
    if grid["cases"]:
        figures += render_grid(grid,output)
    historical_error = "Historical supplied-cone runs are absent from this manual campaign"
    try:
        figures += render_historical(study,output)
    except MissingDataError as error:
        historical_error = str(error)
    by_number = {f["paper_figure"]:f for f in figures}
    coverage = [{"paper_figure":n,"title":title,
        "status":by_number[n].get("status","generated") if n in by_number else historical_error if n == 24 else "unavailable: required saved case or policy absent"}
        for n,title in FIGURES.items()]
    tables.append(write_csv(output/"data/figure_coverage.csv",coverage))
    _report(output,study,figures,coverage,tables)
    changed = [p for p,stamp in study["inputs"].items()
               if (Path(p).stat().st_size,Path(p).stat().st_mtime_ns) != (stamp["size"],stamp["mtime_ns"])]
    if before != _source_hashes() or changed:
        raise RuntimeError("Source or saved data changed during report generation; rerun for a consistent snapshot")
    import numpy, scipy, matplotlib
    manifest = {"input_root":str(source),"input_files":study["inputs"],"issues":study["issues"],
        "simulation_source_sha256":before,"simulation_source_unchanged":True,"inputs_unchanged":True,
        "uses_only_manual_saved_files":True,"figures":figures,"coverage":coverage,"tables":tables,
        "trajectory_stride":args.trajectory_stride,"include_incomplete":args.include_incomplete,
        "software":{"python":platform.python_version(),"numpy":numpy.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
        "outputs":{str(p.relative_to(output)):sha256(p) for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Generated {len(figures)} simulation figures and {len(tables)} CSV tables: {output/'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
