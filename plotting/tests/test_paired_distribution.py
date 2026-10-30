"""Scientific checks for Figure 12's paired sample distributions."""
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pytest

from plotting.common import METRICS,paired_samples,paired_distribution_tables
from plotting import main_plots


def case():
    return {"id":"baseline","raw_selected":{
        "analytical_certificate":{"seeds":[2,1,3],"finals":{metric:[5.,100.,9.] for metric in METRICS}},
        "kracer":{"seeds":[1,3,2],"finals":{metric:[10.,6.,3.] for metric in METRICS}}}}


def test_paired_samples_match_seeds_before_quartiles():
    seeds,values = paired_samples(case(),"analytical_certificate","kracer",METRICS[1])
    assert seeds == [2,1,3]
    np.testing.assert_array_equal(values,[2.,90.,3.])
    summaries,samples = paired_distribution_tables({"cases":{"baseline":case()}})
    assert summaries[0]["median"] == 3
    assert summaries[0]["q1"] == 2.5 and summaries[0]["q3"] == 46.5
    assert {row["seed"]:row["difference"] for row in samples} == {1:90.,2:2.,3:3.}


def test_unpaired_distributions_are_not_invented():
    c = case()
    c["raw_selected"]["kracer"]["seeds"] = [10,30,20]
    assert paired_distribution_tables({"cases":{"baseline":c}}) == ([],[])


def test_endpoint_seed_count_mismatch_rejected():
    c = case()
    c["raw_selected"]["kracer"]["finals"][METRICS[1]] = [2.]
    with pytest.raises(ValueError,match="count mismatch"):
        paired_samples(c,"analytical_certificate","kracer",METRICS[1])


def test_plot_keeps_all_paired_points_and_zero_reference(monkeypatch,tmp_path):
    captured = {}
    def capture(fig,*args,**kwargs):
        captured["figure"] = fig
        return {"paper_figure":12}
    monkeypatch.setattr(main_plots,"save_figure",capture)
    main_plots._paired_ablations({"cases":{"baseline":case()}},tmp_path)
    fig = captured["figure"]
    try:
        ax = fig.axes[0]
        sample_collection = next(c for c in ax.collections
            if isinstance(c,PathCollection) and len(c.get_offsets()) == 3)
        np.testing.assert_array_equal(sample_collection.get_offsets()[:,0],[2.,90.,3.])
        assert any(np.all(np.asarray(line.get_xdata()) == 0) for line in ax.lines)
        assert any("Median 3.00" in text.get_text() and "IQR 44.00" in text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)
