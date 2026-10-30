"""Validate descriptive box statistics and retention of all test observations."""
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pytest

from plotting.common import METRICS, distribution_summary
from plotting import main_plots


def test_quartiles_and_tukey_whiskers_use_raw_regret():
    summary = distribution_summary([1,2,3,4,100])
    assert summary == {"n":5,"mean":22.,"median":3.,"q1":2.,"q3":4.,"iqr":2.,
                       "whisker_low":1.,"whisker_high":4.}


def test_zero_iqr_and_single_observation():
    summary = distribution_summary([0,0,0,0,10])
    assert summary["iqr"] == 0 and summary["mean"] == 2
    assert summary["whisker_low"] == summary["whisker_high"] == 0
    one = distribution_summary([11])
    assert one["n"] == 1 and one["iqr"] == 0
    assert one["mean"] == one["median"] == one["q1"] == one["q3"] == 11


@pytest.mark.parametrize("values",[[],[1,np.nan],[np.inf]])
def test_invalid_distribution_rejected(values):
    with pytest.raises(ValueError):
        distribution_summary(values)


def test_plot_contains_every_sample_including_outlier(monkeypatch,tmp_path):
    values = [1.,2.,3.,4.,100.]
    case = {"id":"baseline","raw_selected":{"kracer":{
        "seeds":[6000,6001,6002,6003,6004],"finals":{metric:values for metric in METRICS}}}}
    captured = {}
    def capture(fig,*args,**kwargs):
        captured["figure"] = fig
        return {"paper_figure":10}
    monkeypatch.setattr(main_plots,"save_figure",capture)
    main_plots._endpoint_plot(case,tmp_path)
    fig = captured["figure"]
    try:
        for ax in fig.axes:
            scatters = [c for c in ax.collections if isinstance(c,PathCollection)]
            samples = next(c for c in scatters if len(c.get_offsets()) == len(values))
            np.testing.assert_array_equal(samples.get_offsets()[:,0],values)
            assert any(len(c.get_offsets()) == 1 and c.get_offsets()[0,0] == 22 for c in scatters)
            assert ax.get_xscale() == "log"
            assert any("Mean 22.00" in text.get_text() and "IQR 2.00" in text.get_text() for text in ax.texts)
    finally:
        plt.close(fig)
