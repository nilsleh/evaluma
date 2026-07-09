import warnings

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

import evaluma
from evaluma.benchmark import Benchmark, BenchmarkGroup
from evaluma.methods.rank_sensitivity import (
    compute_rank_sensitivity,
    compute_rank_sensitivity_from_ranks,
)
from evaluma.results import RankSensitivityResult

matplotlib.use("Agg")


def _make_scores_matrix(scores_dict, datasets=None):
    if datasets is None:
        n = len(next(iter(scores_dict.values())))
        datasets = [f"d{i + 1}" for i in range(n)]
    return pd.DataFrame(scores_dict, index=datasets).T


def _make_bench(scores_dict, datasets=None):
    if datasets is None:
        datasets = ["d1", "d2", "d3", "d4", "d5", "d6"]
    rows = [
        {"model": model, "dataset": d, "metric": "acc", "score": s}
        for model, scores in scores_dict.items()
        for d, s in zip(datasets, scores)
    ]
    return evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )


def _make_group(conditions_to_scores, datasets=None):
    if datasets is None:
        datasets = ["d1", "d2", "d3", "d4", "d5", "d6"]
    rows = []
    for cond, scores_dict in conditions_to_scores.items():
        for model, scores in scores_dict.items():
            for d, s in zip(datasets, scores):
                rows.append(
                    {
                        "optimizer": cond,
                        "model": model,
                        "dataset": d,
                        "metric": "acc",
                        "score": s,
                    }
                )
    df = pd.DataFrame(rows)
    return evaluma.load_df(
        df,
        condition_col="optimizer",
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )


def _make_group_unbounded(conditions_to_scores, datasets=None):
    """Data-driven (``None``-bounds) BenchmarkGroup variant of ``_make_group``."""
    if datasets is None:
        datasets = ["d1", "d2", "d3", "d4", "d5"]
    rows = []
    for cond, scores_dict in conditions_to_scores.items():
        for model, scores in scores_dict.items():
            for d, s in zip(datasets, scores):
                rows.append(
                    {
                        "optimizer": cond,
                        "model": model,
                        "dataset": d,
                        "metric": "acc",
                        "score": s,
                    }
                )
    return evaluma.load_df(
        pd.DataFrame(rows),
        condition_col="optimizer",
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
    )


# C is extreme on every dataset, so under data-driven bounds its presence sets
# the per-dataset max; dropping it re-scales the survivors pre-fix.
_GROUP_ADAM = {
    "A": [0.55, 0.55, 0.55, 0.10, 0.50],
    "B": [0.45, 0.45, 0.45, 0.90, 0.50],
    "C": [1.00, 1.00, 1.00, 0.50, 0.99],
}
_GROUP_SGD = {
    "A": [0.50, 0.50, 0.50, 0.90, 0.55],
    "B": [0.55, 0.55, 0.55, 0.10, 0.45],
    "C": [0.99, 0.99, 0.99, 0.50, 1.00],
}


def test_group_drop_models_preserves_rank_sensitivity():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        group = _make_group_unbounded({"Adam": _GROUP_ADAM, "SGD": _GROUP_SGD})
        keep = ["A", "B"]
        expected = compute_rank_sensitivity(
            group["Adam"].scores_.loc[keep],
            group["SGD"].scores_.loc[keep],
            "Adam",
            "SGD",
            random_state=0,
        )
        actual = group.drop_models(["C"]).rank_sensitivity(
            "Adam", "SGD", random_state=0
        )
        assert actual.tau == pytest.approx(expected.tau)
        assert actual.rho == pytest.approx(expected.rho)
        exp_ranks = expected.table.set_index("model")[
            ["rank_Adam", "rank_SGD"]
        ].sort_index()
        act_ranks = actual.table.set_index("model")[
            ["rank_Adam", "rank_SGD"]
        ].sort_index()
        pd.testing.assert_frame_equal(act_ranks, exp_ranks)


def test_group_drop_datasets_preserves_rank_sensitivity():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        group = _make_group_unbounded({"Adam": _GROUP_ADAM, "SGD": _GROUP_SGD})
        keep_ds = ["d1", "d2", "d3", "d5"]
        expected = compute_rank_sensitivity(
            group["Adam"].scores_[keep_ds],
            group["SGD"].scores_[keep_ds],
            "Adam",
            "SGD",
            random_state=0,
        )
        actual = group.drop_datasets(["d4"]).rank_sensitivity(
            "Adam", "SGD", random_state=0
        )
        assert actual.tau == pytest.approx(expected.tau)
        assert actual.rho == pytest.approx(expected.rho)
        exp_ranks = expected.table.set_index("model")[
            ["rank_Adam", "rank_SGD"]
        ].sort_index()
        act_ranks = actual.table.set_index("model")[
            ["rank_Adam", "rank_SGD"]
        ].sort_index()
        pd.testing.assert_frame_equal(act_ranks, exp_ranks)


def test_default_agg_matches_aggregate_ranking():
    # Model A bombs one task: under mean B ranks top, under trimmed_mean A does.
    scores = {
        "A": [0.9, 0.9, 0.9, 0.9, 0.0],
        "B": [0.8, 0.8, 0.8, 0.8, 0.8],
        "C": [0.5, 0.5, 0.5, 0.5, 0.5],
    }
    datasets = ["d1", "d2", "d3", "d4", "d5"]
    bench = _make_bench(scores, datasets=datasets)
    m = bench.scores_
    result = compute_rank_sensitivity(m, m.copy(), "X", "Y", n_bootstrap=0)
    rank_order = result.table.sort_values("rank_X")["model"].tolist()
    agg_order = bench.aggregate_ranking().table["model"].tolist()
    assert rank_order == agg_order


def test_agg_mean_reproduces_pre_change_tau():
    a = _make_scores_matrix(
        {
            "A": [0.95, 0.20, 0.90, 0.88],
            "B": [0.70, 0.72, 0.71, 0.10],
            "C": [0.65, 0.66, 0.64, 0.67],
            "D": [0.50, 0.55, 0.52, 0.99],
        }
    )
    b = _make_scores_matrix(
        {
            "A": [0.60, 0.61, 0.62, 0.10],
            "B": [0.90, 0.20, 0.88, 0.87],
            "C": [0.70, 0.72, 0.71, 0.69],
            "D": [0.50, 0.55, 0.40, 0.95],
        }
    )
    result = compute_rank_sensitivity(a, b, "Adam", "SGD", agg="mean", random_state=0)
    rank_a = a.mean(axis=1).rank(ascending=False, method="average")
    rank_b = b.mean(axis=1).rank(ascending=False, method="average")
    expected = kendalltau(rank_a.values, rank_b.values, method="auto").statistic
    assert result.tau == pytest.approx(expected)


def test_rank_sensitivity_invalid_agg_raises():
    a = _make_scores_matrix({"A": [0.9, 0.8], "B": [0.8, 0.7]})
    with pytest.raises(ValueError, match="agg"):
        compute_rank_sensitivity(a, a.copy(), "X", "Y", agg="bogus", n_bootstrap=0)


def test_agg_threads_through_group_and_recorded():
    group = _make_group(
        {
            "Adam": {"A": [0.9, 0.9, 0.9, 0.9, 0.0, 0.5], "B": [0.8] * 6},
            "SGD": {"A": [0.8] * 6, "B": [0.9, 0.9, 0.9, 0.9, 0.0, 0.5]},
        }
    )
    r_mean = group.rank_sensitivity("Adam", "SGD", agg="mean", random_state=0)
    assert r_mean.agg == "mean"
    r_default = group.rank_sensitivity("Adam", "SGD", random_state=0)
    assert r_default.agg == "trimmed_mean"


def test_small_n_trimmed_mean_warning():
    a = _make_bench(
        {"A": [0.9, 0.8, 0.7], "B": [0.8, 0.7, 0.6]}, datasets=["d1", "d2", "d3"]
    )
    b = _make_bench(
        {"A": [0.8, 0.9, 0.7], "B": [0.7, 0.8, 0.6]}, datasets=["d1", "d2", "d3"]
    )
    with pytest.warns(UserWarning, match="trim"):
        a.rank_sensitivity(b, "Adam", "SGD", agg="trimmed_mean", random_state=0)
    with pytest.warns(UserWarning) as rec:
        a.rank_sensitivity(b, "Adam", "SGD", agg="mean", random_state=0)
    assert all("trim" not in str(w.message).lower() for w in rec)


def test_perfect_agreement():
    m = _make_scores_matrix(
        {"A": [0.9, 0.8, 0.7], "B": [0.7, 0.6, 0.5], "C": [0.5, 0.4, 0.3]},
        datasets=["d1", "d2", "d3"],
    )
    result = compute_rank_sensitivity(m, m.copy(), "Adam", "SGD", random_state=0)
    assert result.tau == pytest.approx(1.0)
    assert (result.table["delta_rank"] == 0).all()


def test_reversed():
    a = _make_scores_matrix(
        {
            "A": [0.9, 0.9, 0.9, 0.9],
            "B": [0.8, 0.8, 0.8, 0.8],
            "C": [0.7, 0.7, 0.7, 0.7],
            "D": [0.6, 0.6, 0.6, 0.6],
        }
    )
    b = _make_scores_matrix(
        {
            "A": [0.6, 0.6, 0.6, 0.6],
            "B": [0.7, 0.7, 0.7, 0.7],
            "C": [0.8, 0.8, 0.8, 0.8],
            "D": [0.9, 0.9, 0.9, 0.9],
        }
    )
    result = compute_rank_sensitivity(a, b, "Adam", "SGD", random_state=0)
    assert result.tau == pytest.approx(-1.0)


def test_partial_reorder():
    a = _make_scores_matrix(
        {
            "A": [0.95, 0.95, 0.95, 0.95],
            "B": [0.80, 0.80, 0.80, 0.80],
            "C": [0.65, 0.65, 0.65, 0.65],
            "D": [0.50, 0.50, 0.50, 0.50],
        }
    )
    b = _make_scores_matrix(
        {
            "A": [0.92, 0.92, 0.92, 0.92],
            "B": [0.70, 0.70, 0.70, 0.70],
            "C": [0.82, 0.82, 0.82, 0.82],
            "D": [0.55, 0.55, 0.55, 0.55],
        }
    )
    result = compute_rank_sensitivity(a, b, "Adam", "SGD", random_state=0)
    rank_a = a.mean(axis=1).rank(ascending=False, method="average")
    rank_b = b.mean(axis=1).rank(ascending=False, method="average")
    expected = kendalltau(rank_a.values, rank_b.values, method="auto").statistic
    assert result.tau == pytest.approx(expected)


def test_table_schema():
    a = _make_scores_matrix({"A": [0.9, 0.9], "B": [0.8, 0.8], "C": [0.7, 0.7]})
    b = _make_scores_matrix({"A": [0.7, 0.7], "B": [0.9, 0.9], "C": [0.8, 0.8]})
    result = compute_rank_sensitivity(a, b, "Adam", "SGD", random_state=0)
    assert list(result.table.columns) == [
        "model",
        "rank_Adam",
        "rank_SGD",
        "delta_rank",
    ]
    abs_delta = result.table["delta_rank"].abs().tolist()
    assert abs_delta == sorted(abs_delta, reverse=True)


def test_column_names_use_condition_labels():
    a = _make_scores_matrix({"A": [0.9, 0.9], "B": [0.8, 0.8]})
    b = _make_scores_matrix({"A": [0.8, 0.8], "B": [0.9, 0.9]})
    result = compute_rank_sensitivity(a, b, "AdamW", "SGD+Nesterov")
    assert "rank_AdamW" in result.table.columns
    assert "rank_SGD+Nesterov" in result.table.columns


def test_bootstrap_ci_bounds():
    rng = np.random.default_rng(7)
    models = ["A", "B", "C", "D", "E", "F"]
    full_datasets = [f"d{i}" for i in range(10)]
    base_a = np.linspace(0.9, 0.4, num=len(models))
    base_b = np.linspace(0.88, 0.42, num=len(models))

    a10 = pd.DataFrame(
        {
            d: np.clip(base_a + rng.normal(0, 0.08, size=len(models)), 0.0, 1.0)
            for d in full_datasets
        },
        index=models,
    )
    b10 = pd.DataFrame(
        {
            d: np.clip(base_b + rng.normal(0, 0.08, size=len(models)), 0.0, 1.0)
            for d in full_datasets
        },
        index=models,
    )
    a3 = a10.loc[:, full_datasets[:3]]
    b3 = b10.loc[:, full_datasets[:3]]

    r10 = compute_rank_sensitivity(
        a10, b10, "Adam", "SGD", n_bootstrap=300, random_state=0
    )
    r3 = compute_rank_sensitivity(
        a3, b3, "Adam", "SGD", n_bootstrap=300, random_state=0
    )
    assert r10.tau_ci[0] <= r10.tau <= r10.tau_ci[1]
    width10 = r10.tau_ci[1] - r10.tau_ci[0]
    width3 = r3.tau_ci[1] - r3.tau_ci[0]
    assert width10 < width3


def test_n_bootstrap_zero_returns_nan_ci():
    a = _make_scores_matrix({"A": [0.9, 0.8], "B": [0.8, 0.7], "C": [0.7, 0.6]})
    b = _make_scores_matrix({"A": [0.8, 0.9], "B": [0.7, 0.8], "C": [0.6, 0.7]})
    result = compute_rank_sensitivity(a, b, "Adam", "SGD", n_bootstrap=0)
    assert np.isfinite(result.tau)
    assert np.isfinite(result.rho)
    assert np.isnan(result.tau_ci[0]) and np.isnan(result.tau_ci[1])


def test_bootstrap_nan_replicates_handled():
    a = _make_scores_matrix(
        {"A": [0.5, 0.9], "B": [0.5, 0.7], "C": [0.5, 0.6]},
        datasets=["d1", "d2"],
    )
    b = _make_scores_matrix(
        {"A": [0.5, 0.6], "B": [0.5, 0.9], "C": [0.5, 0.7]},
        datasets=["d1", "d2"],
    )
    result = compute_rank_sensitivity(
        a, b, "Adam", "SGD", n_bootstrap=500, random_state=0
    )
    assert isinstance(result.tau_ci, tuple)
    assert len(result.tau_ci) == 2


def test_result_has_rho():
    a = _make_scores_matrix({"A": [0.9, 0.8], "B": [0.8, 0.7], "C": [0.7, 0.6]})
    b = _make_scores_matrix({"A": [0.7, 0.6], "B": [0.8, 0.7], "C": [0.9, 0.8]})
    result = compute_rank_sensitivity(a, b, "Adam", "SGD", random_state=0)
    assert isinstance(result.rho, float)
    assert np.sign(result.rho) == np.sign(result.tau)


def test_table_tie_breaker_model_name():
    a = _make_scores_matrix(
        {"A": [0.9], "B": [0.8], "C": [0.7], "D": [0.6]}, datasets=["d1"]
    )
    b = _make_scores_matrix(
        {"A": [0.6], "B": [0.7], "C": [0.8], "D": [0.9]}, datasets=["d1"]
    )
    result = compute_rank_sensitivity(a, b, "A", "B", n_bootstrap=0)
    abs_delta = result.table["delta_rank"].abs()
    tied = result.table.loc[abs_delta == abs_delta.max(), "model"].tolist()
    assert tied == sorted(tied)


def test_benchmark_method_returns_result():
    a = _make_bench(
        {"A": [0.9, 0.8, 0.7], "B": [0.8, 0.7, 0.6], "C": [0.7, 0.6, 0.5]},
        datasets=["d1", "d2", "d3"],
    )
    b = _make_bench(
        {"A": [0.8, 0.9, 0.7], "B": [0.7, 0.8, 0.6], "C": [0.6, 0.7, 0.5]},
        datasets=["d1", "d2", "d3"],
    )
    result = a.rank_sensitivity(b, "Adam", "SGD", random_state=0)
    assert isinstance(result, RankSensitivityResult)


def test_mismatch_models_raises():
    a = _make_bench({"A": [0.9, 0.8], "B": [0.8, 0.7]}, datasets=["d1", "d2"])
    b = _make_bench(
        {"A": [0.9, 0.8], "B": [0.8, 0.7], "C": [0.7, 0.6]},
        datasets=["d1", "d2"],
    )
    with pytest.raises(ValueError, match="missing from Adam: \\['C'\\]"):
        a.rank_sensitivity(b, "Adam", "SGD")


def test_mismatch_datasets_raises():
    a = _make_bench({"A": [0.9, 0.8], "B": [0.8, 0.7]}, datasets=["d1", "d2"])
    b = _make_bench(
        {"A": [0.9, 0.8, 0.7], "B": [0.8, 0.7, 0.6]}, datasets=["d1", "d2", "d3"]
    )
    with pytest.raises(ValueError, match="missing from Adam: \\['d3'\\]"):
        a.rank_sensitivity(b, "Adam", "SGD")


def test_few_datasets_warns():
    a = _make_bench(
        {"A": [0.9, 0.8, 0.7], "B": [0.8, 0.7, 0.6]}, datasets=["d1", "d2", "d3"]
    )
    b = _make_bench(
        {"A": [0.8, 0.9, 0.7], "B": [0.7, 0.8, 0.6]}, datasets=["d1", "d2", "d3"]
    )
    with pytest.warns(UserWarning, match="bootstrap CI may be wide"):
        a.rank_sensitivity(b, "Adam", "SGD", random_state=0)


def test_random_state_reproducible():
    a = _make_bench(
        {"A": [0.9, 0.8, 0.7], "B": [0.8, 0.7, 0.6]}, datasets=["d1", "d2", "d3"]
    )
    b = _make_bench(
        {"A": [0.8, 0.9, 0.7], "B": [0.7, 0.8, 0.6]}, datasets=["d1", "d2", "d3"]
    )
    r1 = a.rank_sensitivity(b, "Adam", "SGD", random_state=0)
    r2 = a.rank_sensitivity(b, "Adam", "SGD", random_state=0)
    assert r1.tau_ci == r2.tau_ci


def test_load_df_condition_col_returns_benchmark_group():
    group = _make_group(
        {
            "Adam": {"A": [0.9, 0.8], "B": [0.8, 0.7]},
            "SGD": {"A": [0.8, 0.9], "B": [0.7, 0.8]},
        },
        datasets=["d1", "d2"],
    )
    assert isinstance(group, BenchmarkGroup)


def test_load_df_condition_col_requires_at_least_two_conditions():
    rows = [
        {
            "optimizer": "Adam",
            "model": "A",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.9,
        },
        {
            "optimizer": "Adam",
            "model": "B",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.8,
        },
    ]
    with pytest.raises(ValueError, match="at least 2 unique conditions"):
        evaluma.load_df(pd.DataFrame(rows), condition_col="optimizer")


def test_benchmark_group_indexing():
    group = _make_group(
        {
            "Adam": {"A": [0.9, 0.8], "B": [0.8, 0.7]},
            "SGD": {"A": [0.8, 0.9], "B": [0.7, 0.8]},
        },
        datasets=["d1", "d2"],
    )
    assert isinstance(group["Adam"], Benchmark)
    with pytest.raises(KeyError):
        _ = group["unknown"]


def test_benchmark_group_rank_sensitivity_returns_result():
    group = _make_group(
        {
            "Adam": {"A": [0.9, 0.8, 0.7], "B": [0.8, 0.7, 0.6]},
            "SGD": {"A": [0.8, 0.9, 0.7], "B": [0.7, 0.8, 0.6]},
        },
        datasets=["d1", "d2", "d3"],
    )
    result = group.rank_sensitivity("Adam", "SGD", random_state=0)
    assert isinstance(result, RankSensitivityResult)


def test_benchmark_group_rank_sensitivity_same_as_manual():
    adam = {"A": [0.9, 0.8, 0.7], "B": [0.8, 0.7, 0.6]}
    sgd = {"A": [0.8, 0.9, 0.7], "B": [0.7, 0.8, 0.6]}
    group = _make_group({"Adam": adam, "SGD": sgd}, datasets=["d1", "d2", "d3"])
    manual = _make_bench(adam, datasets=["d1", "d2", "d3"]).rank_sensitivity(
        _make_bench(sgd, datasets=["d1", "d2", "d3"]),
        "Adam",
        "SGD",
        random_state=0,
    )
    grouped = group.rank_sensitivity("Adam", "SGD", random_state=0)
    assert grouped.tau == pytest.approx(manual.tau)
    assert grouped.tau_ci == manual.tau_ci
    pd.testing.assert_frame_equal(grouped.table, manual.table)


def test_benchmark_group_rank_sensitivity_requires_explicit_args():
    group = _make_group(
        {
            "Adam": {"A": [0.9, 0.8], "B": [0.8, 0.7]},
            "SGD": {"A": [0.8, 0.9], "B": [0.7, 0.8]},
        },
        datasets=["d1", "d2"],
    )
    with pytest.raises(TypeError):
        group.rank_sensitivity()


def test_benchmark_group_n_conditions():
    group = _make_group(
        {
            "Adam": {"A": [0.9, 0.8], "B": [0.8, 0.7]},
            "SGD": {"A": [0.8, 0.9], "B": [0.7, 0.8]},
            "AdamW": {"A": [0.85, 0.85], "B": [0.75, 0.75]},
        },
        datasets=["d1", "d2"],
    )
    assert len(group._benchmarks) == 3
    assert isinstance(group["AdamW"], Benchmark)


def test_benchmark_group_select_models_returns_group():
    group = _make_group(
        {
            "Adam": {"A": [0.9, 0.8], "B": [0.8, 0.7], "C": [0.7, 0.6]},
            "SGD": {"A": [0.8, 0.9], "B": [0.7, 0.8], "C": [0.6, 0.7]},
        },
        datasets=["d1", "d2"],
    )
    out = group.select_models(["A", "B"])
    assert isinstance(out, BenchmarkGroup)
    assert out["Adam"].scores_.shape[0] == 2
    assert out["SGD"].scores_.shape[0] == 2


def test_benchmark_group_drop_models_returns_group():
    group = _make_group(
        {
            "Adam": {"A": [0.9, 0.8], "B": [0.8, 0.7], "C": [0.7, 0.6]},
            "SGD": {"A": [0.8, 0.9], "B": [0.7, 0.8], "C": [0.6, 0.7]},
        },
        datasets=["d1", "d2"],
    )
    out = group.drop_models(["C"])
    assert out["Adam"].scores_.index.tolist() == ["A", "B"]
    assert out["SGD"].scores_.index.tolist() == ["A", "B"]


def test_benchmark_group_drop_datasets_returns_group():
    group = _make_group(
        {
            "Adam": {"A": [0.9, 0.8], "B": [0.8, 0.7]},
            "SGD": {"A": [0.8, 0.9], "B": [0.7, 0.8]},
        },
        datasets=["D1", "D2"],
    )
    out = group.drop_datasets(["D1"])
    assert out["Adam"].scores_.columns.tolist() == ["D2"]
    assert out["SGD"].scores_.columns.tolist() == ["D2"]


def test_benchmark_drop_models():
    bench = _make_bench(
        {"A": [0.9, 0.8], "B": [0.8, 0.7], "C": [0.7, 0.6]},
        datasets=["d1", "d2"],
    )
    dropped = bench.drop_models(["C"])
    selected = bench.select_models(["A", "B"])
    pd.testing.assert_frame_equal(dropped.scores_, selected.scores_)


def test_benchmark_drop_datasets():
    bench = _make_bench({"A": [0.9, 0.8], "B": [0.8, 0.7]}, datasets=["D1", "D2"])
    dropped = bench.drop_datasets(["D1"])
    selected = bench.select_datasets(["D2"])
    pd.testing.assert_frame_equal(dropped.scores_, selected.scores_)


def test_independent_normalization():
    rows = [
        {
            "optimizer": "Adam",
            "model": "A",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.6,
        },
        {
            "optimizer": "Adam",
            "model": "B",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.2,
        },
        {
            "optimizer": "Adam",
            "model": "C",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.4,
        },
        {
            "optimizer": "SGD",
            "model": "A",
            "dataset": "d1",
            "metric": "acc",
            "score": 1.0,
        },
        {
            "optimizer": "SGD",
            "model": "B",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.0,
        },
        {
            "optimizer": "SGD",
            "model": "C",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.4,
        },
    ]
    group = evaluma.load_df(pd.DataFrame(rows), condition_col="optimizer")
    assert group["Adam"].scores_.loc["C", "d1"] != group["SGD"].scores_.loc["C", "d1"]


def test_load_csv_condition_col(tmp_path):
    rows = [
        {
            "optimizer": "Adam",
            "model": "A",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.9,
        },
        {
            "optimizer": "Adam",
            "model": "B",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.8,
        },
        {
            "optimizer": "SGD",
            "model": "A",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.8,
        },
        {
            "optimizer": "SGD",
            "model": "B",
            "dataset": "d1",
            "metric": "acc",
            "score": 0.9,
        },
    ]
    path = tmp_path / "scores.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    group = evaluma.load_csv(path, condition_col="optimizer")
    assert isinstance(group, BenchmarkGroup)


def test_plot_returns_figure():
    a = _make_scores_matrix({"A": [0.9, 0.8], "B": [0.8, 0.7], "C": [0.7, 0.6]})
    b = _make_scores_matrix({"A": [0.8, 0.9], "B": [0.7, 0.8], "C": [0.6, 0.7]})
    result = compute_rank_sensitivity(a, b, "Adam", "SGD", random_state=0)
    fig = result.plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_no_show(monkeypatch):
    monkeypatch.setattr(
        plt, "show", lambda: (_ for _ in ()).throw(AssertionError("plt.show called"))
    )
    a = _make_scores_matrix({"A": [0.9, 0.8], "B": [0.8, 0.7], "C": [0.7, 0.6]})
    b = _make_scores_matrix({"A": [0.8, 0.9], "B": [0.7, 0.8], "C": [0.6, 0.7]})
    fig = compute_rank_sensitivity(a, b, "Adam", "SGD", random_state=0).plot()
    plt.close(fig)


def test_plot_ax_argument():
    a = _make_scores_matrix({"A": [0.9, 0.8], "B": [0.8, 0.7], "C": [0.7, 0.6]})
    b = _make_scores_matrix({"A": [0.8, 0.9], "B": [0.7, 0.8], "C": [0.6, 0.7]})
    existing_fig, existing_ax = plt.subplots()
    fig = compute_rank_sensitivity(a, b, "Adam", "SGD", random_state=0).plot(
        ax=existing_ax
    )
    assert fig is existing_ax.get_figure()
    plt.close(existing_fig)


# --- from-ranks point-estimate core (decoupled ranker path) ----------------


def test_from_ranks_point_estimate():
    rank_a = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    rank_b = pd.Series({"A": 2.0, "B": 1.0, "C": 4.0, "D": 3.0})
    result = compute_rank_sensitivity_from_ranks(rank_a, rank_b, "cond_a", "cond_b")
    assert isinstance(result, RankSensitivityResult)
    assert -1.0 <= result.tau <= 1.0
    assert -1.0 <= result.rho <= 1.0
    assert np.isnan(result.tau_ci[0]) and np.isnan(result.tau_ci[1])
    assert set(result.table.columns) == {
        "model",
        "rank_cond_a",
        "rank_cond_b",
        "delta_rank",
    }
    abs_delta = result.table["delta_rank"].abs().tolist()
    assert abs_delta == sorted(abs_delta, reverse=True)


def test_from_ranks_realigns_b_to_a():
    rank_a = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    rank_b = pd.Series({"A": 2.0, "B": 1.0, "C": 4.0, "D": 3.0})
    shuffled = rank_b.loc[["D", "B", "A", "C"]]
    ref = compute_rank_sensitivity_from_ranks(rank_a, rank_b, "X", "Y")
    out = compute_rank_sensitivity_from_ranks(rank_a, shuffled, "X", "Y")
    assert out.tau == pytest.approx(ref.tau)
    assert out.rho == pytest.approx(ref.rho)
    pd.testing.assert_frame_equal(out.table, ref.table)


def test_from_ranks_constant_vector_nan():
    rank_a = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    rank_b = pd.Series({"A": 2.0, "B": 2.0, "C": 2.0})  # degenerate / all tied
    result = compute_rank_sensitivity_from_ranks(rank_a, rank_b, "X", "Y")
    assert np.isnan(result.tau)


# --- ranker registry (Benchmark._rank_vector) ------------------------------


_RANKER_SCORES = {
    "A": [0.95, 0.90, 0.92, 0.88, 0.94, 0.91],
    "B": [0.80, 0.85, 0.82, 0.78, 0.81, 0.83],
    "C": [0.70, 0.65, 0.72, 0.68, 0.71, 0.69],
    "D": [0.55, 0.60, 0.52, 0.58, 0.54, 0.61],
}


@pytest.mark.parametrize("ranker", ["avg_rank", "elo", "improvability"])
def test_rank_vector_named(ranker):
    bench = _make_bench(_RANKER_SCORES)
    ranks = bench._rank_vector(ranker)
    assert isinstance(ranks, pd.Series)
    assert set(ranks.index) == set(_RANKER_SCORES)
    assert not ranks.isna().any()


def test_rank_vector_callable_passthrough():
    bench = _make_bench(_RANKER_SCORES)
    sentinel = pd.Series({m: i for i, m in enumerate(_RANKER_SCORES)})

    def custom(_bench):
        return sentinel

    out = bench._rank_vector(custom)
    pd.testing.assert_series_equal(out, sentinel.astype(float))


def test_rank_vector_callable_requires_series():
    bench = _make_bench(_RANKER_SCORES)

    def custom(_bench):
        return {"A": 1.0}

    with pytest.raises(TypeError, match="pandas Series"):
        bench._rank_vector(custom)


def test_rank_vector_callable_requires_matching_models():
    bench = _make_bench(_RANKER_SCORES)

    def custom(_bench):
        return pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})

    with pytest.raises(ValueError, match="missing models"):
        bench._rank_vector(custom)


def test_rank_vector_callable_requires_numeric_finite_values():
    bench = _make_bench(_RANKER_SCORES)

    def non_numeric(_bench):
        return pd.Series(
            {m: f"rank-{i}" for i, m in enumerate(_RANKER_SCORES, start=1)}
        )

    with pytest.raises(TypeError, match="numeric rank values"):
        bench._rank_vector(non_numeric)

    def non_finite(_bench):
        return pd.Series({m: np.inf for m in _RANKER_SCORES})

    with pytest.raises(ValueError, match="finite rank values"):
        bench._rank_vector(non_finite)


def test_rank_vector_unknown_raises():
    bench = _make_bench(_RANKER_SCORES)
    with pytest.raises(ValueError, match="avg_rank.*elo.*improvability"):
        bench._rank_vector("bogus")


# --- Benchmark.rank_sensitivity(ranker=...) wiring -------------------------


_RANKER_SCORES_B = {
    "A": [0.80, 0.85, 0.82, 0.78, 0.81, 0.83],
    "B": [0.95, 0.90, 0.92, 0.88, 0.94, 0.91],
    "C": [0.55, 0.60, 0.52, 0.58, 0.54, 0.61],
    "D": [0.70, 0.65, 0.72, 0.68, 0.71, 0.69],
}


def test_aggregate_backward_compat():
    a = _make_bench(_RANKER_SCORES)
    b = _make_bench(_RANKER_SCORES_B)
    got = a.rank_sensitivity(b, "condA", "condB", ranker="aggregate", random_state=0)
    aligned_b = b.scores_.loc[a.scores_.index, a.scores_.columns]
    expected = compute_rank_sensitivity(
        a.scores_, aligned_b, "condA", "condB", random_state=0
    )
    assert got.tau == pytest.approx(expected.tau)
    assert got.tau_ci == pytest.approx(expected.tau_ci)
    assert got.rho == pytest.approx(expected.rho)
    pd.testing.assert_frame_equal(got.table, expected.table)


@pytest.mark.parametrize("ranker", ["avg_rank", "elo", "improvability"])
def test_ranker_point_estimate(ranker):
    a = _make_bench(_RANKER_SCORES)
    b = _make_bench(_RANKER_SCORES_B)
    result = a.rank_sensitivity(b, "condA", "condB", ranker=ranker)
    assert -1.0 <= result.tau <= 1.0
    assert np.isnan(result.tau_ci[0]) and np.isnan(result.tau_ci[1])


def test_ranker_callable():
    a = _make_bench(_RANKER_SCORES)
    b = _make_bench(_RANKER_SCORES_B)

    def by_mean(bench):
        return bench.scores_.mean(axis=1).rank(ascending=False, method="average")

    result = a.rank_sensitivity(b, "condA", "condB", ranker=by_mean)
    assert -1.0 <= result.tau <= 1.0
    assert np.isnan(result.tau_ci[0]) and np.isnan(result.tau_ci[1])


@pytest.mark.parametrize("ranker", ["avg_rank", "elo", "improvability"])
def test_group_ranker_point_estimate(ranker):
    group = _make_group({"condA": _RANKER_SCORES, "condB": _RANKER_SCORES_B})
    result = group.rank_sensitivity("condA", "condB", ranker=ranker)
    assert -1.0 <= result.tau <= 1.0
    assert np.isnan(result.tau_ci[0]) and np.isnan(result.tau_ci[1])
    assert result.agg == ranker


def test_ranker_warns_on_bootstrap():
    a = _make_bench(_RANKER_SCORES)
    b = _make_bench(_RANKER_SCORES_B)
    with pytest.warns(UserWarning, match="bootstrap|CI|ignored"):
        a.rank_sensitivity(b, "condA", "condB", ranker="elo", n_bootstrap=100)


def test_ranker_validates_identical_sets():
    a = _make_bench(_RANKER_SCORES)
    b_missing = _make_bench({m: v for m, v in _RANKER_SCORES_B.items() if m != "D"})
    with pytest.raises(ValueError, match="mismatch"):
        a.rank_sensitivity(b_missing, "condA", "condB", ranker="elo")
