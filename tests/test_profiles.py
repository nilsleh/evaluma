import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import evaluma

matplotlib.use("Agg")

# Scores: A always best, ratios = best/score for max direction
# best per dataset: A's scores (0.9, 0.8, 0.7, 0.6)
EXPECTED_RATIOS = {
    ("A", "d1"): 1.0,
    ("A", "d2"): 1.0,
    ("A", "d3"): 1.0,
    ("A", "d4"): 1.0,
    ("B", "d1"): 0.9 / 0.5,
    ("B", "d2"): 0.8 / 0.6,
    ("B", "d3"): 0.7 / 0.4,
    ("B", "d4"): 0.6 / 0.5,
    ("C", "d1"): 0.9 / 0.2,
    ("C", "d2"): 0.8 / 0.3,
    ("C", "d3"): 0.7 / 0.25,
    ("C", "d4"): 0.6 / 0.35,
}


def test_profiles_best_model_fraction_one_at_tau_one(bench):
    result = bench.performance_profiles()
    min_tau = result.table["tau"].min()
    assert abs(min_tau - 1.0) < 1e-12
    a_frac = result.table[
        (result.table["tau"] == min_tau) & (result.table["model"] == "A")
    ].iloc[0]["fraction_within_tau"]
    assert abs(a_frac - 1.0) < 1e-9


def test_profiles_exact_breakpoints(bench):
    # tau grid must contain every observed ratio as an exact value
    result = bench.performance_profiles()
    observed_taus = set(result.table["tau"].unique())
    for (model, dataset), expected_ratio in EXPECTED_RATIOS.items():
        assert any(
            abs(t - expected_ratio) < 1e-9 for t in observed_taus
        ), f"ratio {expected_ratio:.4f} for ({model},{dataset}) missing from tau grid"


def test_profiles_table_schema(bench):
    result = bench.performance_profiles()
    assert result.table.columns.tolist() == ["tau", "model", "fraction_within_tau"]


def test_profiles_aup_attribute(bench):
    result = bench.performance_profiles()
    assert hasattr(result, "aup")
    aup = result.aup
    assert isinstance(aup, pd.Series)
    assert len(aup) == len(result.table["model"].unique())
    assert (aup >= 0).all()
    assert aup.notna().all()


def test_profiles_plot_returns_figure(bench):
    fig = bench.performance_profiles().plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_profiles_plot_one_line_per_model(bench):
    fig = bench.performance_profiles().plot()
    ax = fig.axes[0]
    assert len(ax.lines) == 3
    plt.close(fig)


def test_profiles_plot_xaxis_is_log10(bench):
    result = bench.performance_profiles()
    fig = result.plot()
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    # x values are raw tau ratios; all must be >= 1
    for line in ax.lines:
        assert (np.array(line.get_xdata()) >= 1).all()
    plt.close(fig)


def test_profiles_zero_score_raises():
    rows = [
        {"model": "A", "dataset": "d1", "metric": "acc", "score": 0.9},
        {"model": "B", "dataset": "d1", "metric": "acc", "score": 0.0},
    ]
    bench = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    with pytest.raises(ValueError, match="score of 0 or below"):
        bench.performance_profiles()


def test_profiles_max_tau_all_reach_one(bench):
    result = bench.performance_profiles()
    max_tau = result.table["tau"].max()
    at_max = result.table[result.table["tau"] == max_tau]
    assert np.allclose(at_max["fraction_within_tau"].values, 1.0, atol=1e-9)


def test_profiles_min_direction_correct_ratios():
    """For a min-direction metric, ratio = score / best (= score / min_score)."""
    # Model A: RMSE 0.5 (best), Model B: RMSE 1.0 — correct ratio B/A = 2.0
    rows = [
        {"model": "A", "dataset": "rmse_task", "metric": "rmse", "score": 0.5},
        {"model": "B", "dataset": "rmse_task", "metric": "rmse", "score": 1.0},
    ]
    bench = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=2.0,
        metric_direction={"rmse_task": "min"},
    )
    result = bench.performance_profiles()
    taus = result.table["tau"].unique()
    # Expected taus: 1.0 (A is best) and 2.0 (B's ratio = 1.0/0.5)
    assert 1.0 in taus
    assert any(abs(t - 2.0) < 1e-9 for t in taus)
    # At tau=1.0, only A has ratio <= 1 → fraction = 0.5 for each
    at_1 = result.table[result.table["tau"] == 1.0]
    a_frac = at_1[at_1["model"] == "A"].iloc[0]["fraction_within_tau"]
    b_frac = at_1[at_1["model"] == "B"].iloc[0]["fraction_within_tau"]
    assert abs(a_frac - 1.0) < 1e-9
    assert abs(b_frac - 0.0) < 1e-9


def test_profiles_min_direction_not_inflated():
    """Verify that min-direction ratios use score/min, not (1-score)/(1-min)."""
    # RMSE values close together: correct max ratio ≈ 1.009, not ~1.18
    rows = [
        {"model": m, "dataset": "bio", "metric": "rmse", "score": s}
        for m, s in [("best", 0.948), ("worst", 0.957)]
    ]
    bench = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
        metric_direction={"bio": "min"},
    )
    result = bench.performance_profiles()
    max_tau = result.table["tau"].max()
    expected = 0.957 / 0.948
    assert abs(max_tau - expected) < 1e-9, (
        f"max tau {max_tau:.6f} != expected {expected:.6f}; "
        "normalization-inflated ratios detected"
    )


def test_profiles_tied_profiles_coincide():
    rows = [
        {"model": m, "dataset": d, "metric": "acc", "score": 0.7}
        for m in ["X", "Y"]
        for d in ["d1", "d2", "d3", "d4"]
    ]
    b = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    result = b.performance_profiles()
    for tau in result.table["tau"].unique():
        at_tau = result.table[result.table["tau"] == tau]
        fracs = at_tau["fraction_within_tau"].values
        assert np.allclose(fracs, fracs[0], atol=1e-9)


def test_profiles_single_dataset_best_ratio_one():
    rows = [
        {"model": "A", "dataset": "d1", "metric": "acc", "score": 0.9},
        {"model": "B", "dataset": "d1", "metric": "acc", "score": 0.5},
    ]
    b = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    result = b.performance_profiles()
    min_tau = result.table["tau"].min()
    a_frac = result.table[
        (result.table["tau"] == min_tau) & (result.table["model"] == "A")
    ].iloc[0]["fraction_within_tau"]
    assert abs(a_frac - 1.0) < 1e-9


def test_profiles_all_perfect_fraction_one_at_tau_one():
    rows = [
        {"model": m, "dataset": f"d{i}", "metric": "acc", "score": 1.0}
        for m in ["A", "B"]
        for i in range(4)
    ]
    b = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    result = b.performance_profiles()
    at_one = result.table[np.isclose(result.table["tau"], 1.0)]
    assert (at_one["fraction_within_tau"] == 1.0).all()


def test_profiles_aup_best_model_wins():
    rows = (
        [{"model": "Best", "dataset": f"d{i}", "metric": "acc", "score": 0.9} for i in range(1, 6)]
        + [{"model": "Worse", "dataset": f"d{i}", "metric": "acc", "score": 0.5} for i in range(1, 6)]
    )
    b = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    aup = b.performance_profiles().aup
    assert aup["Best"] > aup["Worse"]


def test_profiles_aup_nonnegative(bench):
    aup = bench.performance_profiles().aup
    assert (aup >= 0).all()
    assert aup.notna().all()
