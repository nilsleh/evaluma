import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import evaluma

matplotlib.use("Agg")

EXPECTED_SCORES = {"A": 0.75, "B": 0.5, "C": 0.275}


def test_aggregate_table_schema(bench):
    result = bench.aggregate_ranking()
    assert result.table.columns.tolist() == ["model", "score"]


def test_aggregate_trimmed_mean_exact(bench):
    result = bench.aggregate_ranking(agg="trimmed_mean")
    computed = result.table.set_index("model")["score"].to_dict()
    for model, expected in EXPECTED_SCORES.items():
        assert abs(computed[model] - expected) < 1e-6


def test_aggregate_mean_exact(bench):
    result = bench.aggregate_ranking(agg="mean")
    computed = result.table.set_index("model")["score"].to_dict()
    for model, expected in EXPECTED_SCORES.items():
        assert abs(computed[model] - expected) < 1e-9


def test_aggregate_median_exact(bench):
    result = bench.aggregate_ranking(agg="median")
    computed = result.table.set_index("model")["score"].to_dict()
    for model, expected in EXPECTED_SCORES.items():
        assert abs(computed[model] - expected) < 1e-9


def test_aggregate_sorted_descending(bench):
    result = bench.aggregate_ranking()
    assert result.table["score"].is_monotonic_decreasing


def test_aggregate_no_ci_columns(bench):
    result = bench.aggregate_ranking()
    assert "CI_low" not in result.table.columns
    assert "CI_high" not in result.table.columns


def test_aggregate_plot_returns_figure(bench):
    fig = bench.aggregate_ranking().plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_aggregate_works_with_seed_bench(bench_runs, bench):
    result_runs = bench_runs.aggregate_ranking()
    result_plain = bench.aggregate_ranking()
    for model in ["A", "B", "C"]:
        val_runs = result_runs.table.set_index("model").loc[model, "score"]
        val_plain = result_plain.table.set_index("model").loc[model, "score"]
        assert abs(val_runs - val_plain) < 1e-9


def test_aggregate_invalid_agg(bench):
    with pytest.raises(ValueError, match="agg"):
        bench.aggregate_ranking(agg="something_invalid")


# --- ported from test_iqm.py (trimmed-mean arithmetic) ---

def test_aggregate_middle_four_equals_mean():
    scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    rows = [
        {"model": "M", "dataset": f"d{i}", "metric": "acc", "score": s}
        for i, s in enumerate(scores)
    ]
    b = evaluma.load_df(
        pd.DataFrame(rows), model="model", dataset="dataset", metric="metric", score="score",
        norm_ref_low=0.0, norm_ref_high=1.0,
    )
    result = b.aggregate_ranking(agg="trimmed_mean")
    expected = np.mean(sorted(scores)[2:6])
    assert abs(result.table.iloc[0]["score"] - expected) < 1e-9


def test_aggregate_outlier_stability():
    base_scores = [0.5, 0.6, 0.7, 0.8]
    rows_base = [
        {"model": "M", "dataset": f"d{i}", "metric": "acc", "score": s}
        for i, s in enumerate(base_scores)
    ]
    rows_outlier = rows_base + [
        {"model": "M", "dataset": "d4", "metric": "acc", "score": 0.01},
        {"model": "M", "dataset": "d5", "metric": "acc", "score": 0.99},
    ]
    b_base = evaluma.load_df(
        pd.DataFrame(rows_base), model="model", dataset="dataset", metric="metric", score="score",
        norm_ref_low=0.0, norm_ref_high=1.0,
    )
    b_out = evaluma.load_df(
        pd.DataFrame(rows_outlier), model="model", dataset="dataset", metric="metric", score="score",
        norm_ref_low=0.0, norm_ref_high=1.0,
    )
    score_base = b_base.aggregate_ranking(agg="trimmed_mean").table.iloc[0]["score"]
    score_out = b_out.aggregate_ranking(agg="trimmed_mean").table.iloc[0]["score"]
    assert abs(score_base - score_out) < 1e-9


def test_aggregate_tied_models_equal():
    rows = [
        {"model": m, "dataset": d, "metric": "acc", "score": 0.7}
        for m in ["X", "Y"]
        for d in ["d1", "d2", "d3", "d4"]
    ]
    b = evaluma.load_df(
        pd.DataFrame(rows), model="model", dataset="dataset", metric="metric", score="score",
        norm_ref_low=0.0, norm_ref_high=1.0,
    )
    result = b.aggregate_ranking(agg="trimmed_mean")
    scores = result.table.set_index("model")["score"]
    assert abs(scores["X"] - scores["Y"]) < 1e-9


def test_aggregate_single_dataset():
    rows = [{"model": "M", "dataset": "d1", "metric": "acc", "score": 0.65}]
    b = evaluma.load_df(
        pd.DataFrame(rows), model="model", dataset="dataset", metric="metric", score="score",
        norm_ref_low=0.0, norm_ref_high=1.0,
    )
    result = b.aggregate_ranking(agg="trimmed_mean")
    assert abs(result.table.iloc[0]["score"] - 0.65) < 1e-9


def test_aggregate_all_perfect():
    rows = [
        {"model": "M", "dataset": f"d{i}", "metric": "acc", "score": 1.0}
        for i in range(4)
    ]
    b = evaluma.load_df(
        pd.DataFrame(rows), model="model", dataset="dataset", metric="metric", score="score",
        norm_ref_low=0.0, norm_ref_high=1.0,
    )
    result = b.aggregate_ranking(agg="trimmed_mean")
    assert abs(result.table.iloc[0]["score"] - 1.0) < 1e-9
