"""Validate additional-data collection and paper-equivalent curve semantics."""
from collections import Counter
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from plotting.collect_additional import budget_episode,historical_episode,_native
from plotting.common import METRICS
from plotting.additional import curve_stats
from plotting import grid_plots


PROJECT = Path(__file__).resolve().parents[2]


def test_both_metrics_match_independent_full_runs():
    project = PROJECT
    if not (project/"src").exists():
        project = Path(r"D:\comp_courses\comp3770\K-RACER\manual_simulations")
    cfg = {"D":3,"K":3,"T":100,"sigma_u":.1/np.sqrt(3),"gamma_cara":1.,"delta_conf":.05}
    job = {"project":str(project),"config":cfg,"seed":6000,"gamma":.9,"budgets":[0,1,4,16,32]}
    collected = budget_episode(job)
    engine,_,Sim,User,Rec = _native(project)
    for row in collected:
        sim = Sim(ambient_dimension=3,num_steps=100,seed=6000,user_config=User(sigma_u=cfg["sigma_u"],risk=1.))
        rec = Rec(slate_size=3,cutoff=row["T0"],gamma_ctr=.9)
        result = engine.run_simulation(sim,rec,"kracer",cutoff_after_startup=True)
        consumed = sum(1.-float(i.utility) for i in result.interactions)
        slate = sum(1.-sum(float(result.user.user_state.preference@item.emb) for item in i.slate)/len(i.slate) for i in result.interactions)
        np.testing.assert_allclose([row["consumed_regret"],row["slate_average_regret"]],[consumed,slate],atol=1e-9,rtol=1e-10)
        assert row["phase_counts"] == dict(Counter(i.phase.name.lower() for i in result.interactions))


def test_historical_truth_pairing_and_fixed_exploration():
    project = PROJECT if (PROJECT/"src").exists() else Path(r"D:\comp_courses\comp3770\K-RACER\manual_simulations")
    job = {"project":str(project),"model":"bayesian","seed":26000,"horizon":100,
           "noise":.1,"risk":1.,"delta_conf":.05,"K":2}
    a = historical_episode(job)
    b = historical_episode(dict(job,K=3,risk=-5.))
    exact = historical_episode(dict(job,model="perfect"))
    np.testing.assert_array_equal(a["preference"],b["preference"])
    np.testing.assert_array_equal(a["preference"],exact["preference"])
    assert a["preference"][0] >= 1/np.sqrt(2)
    assert sum(a["phase_counts"].get(p,0) for p in ("excite","localize")) == 64
    assert sum(a["phase_counts"].values()) == 100
    assert "startup" not in a["phase_counts"]


def test_curve_stats_requires_saved_test_observations():
    with pytest.raises(ValueError,match="not saved"):
        curve_stats({},METRICS[1])
    curves = {0:{METRICS[1]:[4.,6.]},4:{METRICS[1]:[8.,10.]}}
    budgets,means,lo,hi = curve_stats({"budget_sweep":curves},METRICS[1])
    np.testing.assert_array_equal(budgets,[0,4])
    np.testing.assert_array_equal(means,[5,9])
    assert np.all(lo<means) and np.all(hi>means)


def test_figure20_plots_only_slate_metric_and_keeps_validation_marker(monkeypatch,tmp_path):
    case = {"id":"baseline","config":{"D":5,"K":2,"sigma_u":.1/np.sqrt(5)},
        "selected_T0":{"kracer":4},"raw_selected":{"kracer":{}},
        "budget_sweep":{0:{METRICS[0]:[200,300],METRICS[1]:[4,6]},
                        4:{METRICS[0]:[400,500],METRICS[1]:[8,10]}}}
    saved = {}
    def capture(fig,*args,**kwargs):
        saved["fig"] = fig
        return {"paper_figure":20}
    monkeypatch.setattr(grid_plots,"save_figure",capture)
    grid_plots.plot_grid_cutoff_curves({"cases":{"baseline":case}},tmp_path)
    fig = saved["fig"]
    try:
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.lines[0].get_ydata(),[5.,9.])
        np.testing.assert_array_equal(ax.lines[1].get_xdata(),[4])
        assert "slate-average" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_full_collected_data_and_fixed_gamma_when_available(tmp_path):
    from plotting.additional import cached_study,attach_supplement
    project = PROJECT if (PROJECT/"src").exists() else Path(r"D:\comp_courses\comp3770\K-RACER\manual_simulations")
    report = project/"plotting/results"
    supplement = report/"additional/supplement.json"
    if not supplement.exists():
        pytest.skip("Additional campaign is not collected yet")
    study,_ = cached_study(report)
    attach_supplement(study,supplement,report)
    assert sum(bool(c.get("budget_sweep")) for c in study["cases"].values()) == 12
    assert all(len(c["budget_sweep"]) == 11 for c in study["cases"].values() if c.get("budget_sweep"))
    assert "radius_0.25" in study["cases"] and "radius_0.75" in study["cases"]
    assert len(study["historical_records"]) == 180
    bad = json.loads(supplement.read_text())
    next(r for r in bad["records"] if r["job"]["kind"] == "budget")["job"]["gamma"] += .01
    path = tmp_path/"wrong_gamma.json"
    path.write_text(json.dumps(bad))
    original,_ = cached_study(report)
    with pytest.raises(ValueError,match="gamma"):
        attach_supplement(original,path,report)
