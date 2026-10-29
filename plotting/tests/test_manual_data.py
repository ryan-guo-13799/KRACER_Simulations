"""Checks for the native manual JSON format, without importing simulation code."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from plotting.common import METRICS, mean_ci, paired, phase_values
from plotting.data import MissingDataError, export_csvs, grid_view, load_run


def episode(algorithm="kracer",seed=6000,case="baseline",cost=.25,noise=.1,K=2):
    phases = ["STARTUP","EXCITE","LOCALIZE","EXPLOIT"]
    return {"case":case,"algorithm":algorithm,"label":case,"seed":seed,
        "settings":{"dimension":5,"slate_size":K,"horizon":4,"post_startup_cutoff":2,
            "absolute_cutoff":3,"sigma_u":noise/np.sqrt(5),"normalized_noise":noise,"risk":1.,
            "delta_conf":.05,"gamma_ctr":.9,"analytical_gate_size":4.,"startup_half_angle_radians":1.3},
        "preference":[1.,0.,0.,0.,0.],"startup":{"rounds":1,"status":"CERTIFIED"},
        "summary":{"phase_counts":{p:1 for p in phases},"consumed_regret":cost*4,"slate_average_regret":cost*8},
        "rounds":[{"t":i+1,"phase":p,"consumed_regret":cost,"slate_average_regret":cost*2} for i,p in enumerate(phases)]}


def save_group(root,algorithm="kracer",case="baseline",seeds=(6000,6001),parallel=False,**kwargs):
    directory = root/algorithm/case/algorithm if parallel else root/case/algorithm
    directory.mkdir(parents=True,exist_ok=True)
    validation = {"selected_cutoff":2,"parameter_name":"gamma_ctr","selected_parameter":.9,
        "validation_seeds":[120000,120001],"test_seeds":list(seeds),
        "scores":[{"parameter":.9,"cutoff":0,"seed_scores":[4.,6.],"mean_slate_regret":5.},
                  {"parameter":.9,"cutoff":2,"seed_scores":[1.,3.],"mean_slate_regret":2.}]}
    (directory/"validation.json").write_text(json.dumps(validation))
    for seed in seeds:
        (directory/f"seed_{seed}.json").write_text(json.dumps(episode(algorithm,seed,case,**kwargs)))
    return directory


@pytest.mark.parametrize("parallel",[True,False])
def test_reads_native_layout_and_accounts_phases(tmp_path,parallel):
    save_group(tmp_path,parallel=parallel)
    study = load_run(tmp_path)
    c = study["cases"]["baseline"]
    assert c["keys"] == ["kracer"]
    assert c["selected_T0"]["kracer"] == 2
    assert c["statistics"]["kracer"][METRICS[0]]["mean"] == 1.
    assert c["trajectories"]["kracer"][METRICS[0]]["mean"][-1] == 1.
    assert sum(v.mean() for v in phase_values(c,"kracer",METRICS[1]).values()) == 2.
    assert study["issues"] == []


def test_pairing_uses_seed_ids(tmp_path):
    save_group(tmp_path)
    save_group(tmp_path,"minimum_eigenvector",seeds=(6001,6000),cost=.5)
    case = load_run(tmp_path)["cases"]["baseline"]
    assert paired(case,"minimum_eigenvector","kracer",METRICS[0])["mean"] == 1.


def test_incomplete_excluded_by_default_and_opt_in(tmp_path):
    save_group(tmp_path)
    directory = save_group(tmp_path,"minimum_eigenvector")
    (directory/"seed_6001.json").unlink()
    study = load_run(tmp_path)
    assert study["cases"]["baseline"]["keys"] == ["kracer"]
    assert any(r["status"] == "incomplete group" for r in study["issues"])
    study = load_run(tmp_path,include_incomplete=True)
    case = study["cases"]["baseline"]
    stats = case["statistics"]["minimum_eigenvector"][METRICS[0]]
    assert stats["n"] == 1 and stats["lo"] is None
    with pytest.raises(ValueError,match="seed sets differ"):
        paired(case,"minimum_eigenvector","kracer",METRICS[0])


@pytest.mark.parametrize("mutation",["wrong_total","wrong_phase","wrong_selection","leakage"])
def test_rejects_inconsistent_native_records(tmp_path,mutation):
    directory = save_group(tmp_path,seeds=(6000,))
    name = "validation.json" if mutation in ("wrong_selection","leakage") else "seed_6000.json"
    path = directory/name
    payload = json.loads(path.read_text())
    if mutation == "wrong_total": payload["summary"]["consumed_regret"] = 99
    if mutation == "wrong_phase": payload["summary"]["phase_counts"]["STARTUP"] = 99
    if mutation == "wrong_selection": payload["selected_cutoff"] = 0
    if mutation == "leakage": payload["validation_seeds"][0] = 6000
    path.write_text(json.dumps(payload))
    with pytest.raises(MissingDataError): load_run(tmp_path)


def test_grid_uses_actual_manual_noise_and_excludes_dimensions(tmp_path):
    save_group(tmp_path)
    save_group(tmp_path,case="speed_0.15",noise=.15)
    save_group(tmp_path,case="K_9",K=9)
    view = grid_view(load_run(tmp_path))
    assert set(view["cases"]) == {"baseline","speed_0.15","K_9"}
    cfg = view["cases"]["speed_0.15"]["config"]
    assert cfg["sigma_u"]*np.sqrt(cfg["D"]) == pytest.approx(.15)


def test_exports_preserve_input_and_validation_split(tmp_path):
    source,output = tmp_path/"saved",tmp_path/"report"
    save_group(source)
    before = {str(p):p.read_bytes() for p in source.rglob("*.json")}
    study = load_run(source)
    copy = deepcopy(study)
    tables = export_csvs(study,output)
    assert len(tables) == 9
    assert study == copy
    assert before == {str(p):p.read_bytes() for p in source.rglob("*.json")}
    assert "validation" in (output/"data/validation_scores.csv").read_text()
    assert "test_consumed_means" not in str(study)


def test_single_seed_interval_is_unavailable():
    assert mean_ci([2.])["lo"] is None


def test_no_foreign_data_or_simulation_imports():
    import ast
    root = Path(__file__).resolve().parents[1]
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or "").startswith(("src","simulation","theory_plots"))
            if isinstance(node,ast.Import):
                assert all(not name.name.startswith(("src","simulation")) for name in node.names)
