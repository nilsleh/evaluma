import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import evaluma
from evaluma.methods.improvability import _compute_error_matrix, compute_improvability

matplotlib.use("Agg")

# --- Slice 2: error reconstruction + improvability compute ---


def test_error_matrix_max_direction():
    raw = pd.DataFrame({"d1": [0.9, 0.5], "d2": [0.8, 0.6]}, index=["A", "B"])
    direction_map = {"d1": "max", "d2": "max"}
    optimum_map = {"d1": 1.0, "d2": 1.0}
    err = _compute_error_matrix(raw, direction_map, optimum_map)
    expected = pd.DataFrame({"d1": [0.1, 0.5], "d2": [0.2, 0.4]}, index=["A", "B"])
    assert err.index.tolist() == raw.index.tolist()
    assert err.columns.tolist() == raw.columns.tolist()
    assert err.shape == raw.shape
    assert np.abs((err - expected).to_numpy()).max() < 1e-9


def test_error_matrix_min_direction():
    raw = pd.DataFrame({"d1": [0.9, 0.5], "d2": [0.8, 0.6]}, index=["A", "B"])
    direction_map = {"d1": "min", "d2": "min"}
    optimum_map = {"d1": 0.0, "d2": 0.0}
    err = _compute_error_matrix(raw, direction_map, optimum_map)
    # score - optimum == score
    assert np.abs((err - raw).to_numpy()).max() < 1e-9


def test_error_matrix_mixed_direction():
    raw = pd.DataFrame({"d1": [0.9, 0.5], "d2": [0.8, 0.6]}, index=["A", "B"])
    direction_map = {"d1": "max", "d2": "min"}
    optimum_map = {"d1": 1.0, "d2": 0.0}
    err = _compute_error_matrix(raw, direction_map, optimum_map)
    expected = pd.DataFrame({"d1": [0.1, 0.5], "d2": [0.8, 0.6]}, index=["A", "B"])
    assert np.abs((err - expected).to_numpy()).max() < 1e-9


def _max_result():
    raw = pd.DataFrame({"d1": [0.9, 0.5], "d2": [0.8, 0.6]}, index=["A", "B"])
    return compute_improvability(
        raw, {"d1": "max", "d2": "max"}, {"d1": 1.0, "d2": 1.0}
    )


def test_improvability_exact_max():
    result = _max_result()
    computed = result.table.set_index("model")["improvability"].to_dict()
    # errors A:(0.1,0.2) B:(0.5,0.4); best d1=0.1 d2=0.2
    # A: mean(0, 0) = 0 ; B: mean(80, 50) = 65
    assert abs(computed["A"] - 0.0) < 1e-9
    assert abs(computed["B"] - 65.0) < 1e-9


def test_improvability_exact_min():
    raw = pd.DataFrame({"d1": [0.1, 0.5], "d2": [0.2, 0.4]}, index=["A", "B"])
    result = compute_improvability(
        raw, {"d1": "min", "d2": "min"}, {"d1": 0.0, "d2": 0.0}
    )
    computed = result.table.set_index("model")["improvability"].to_dict()
    assert abs(computed["A"] - 0.0) < 1e-9
    assert abs(computed["B"] - 65.0) < 1e-9


def test_improvability_best_model_zero():
    result = _max_result()
    computed = result.table.set_index("model")["improvability"].to_dict()
    # A is best on every dataset -> overall improvability 0
    assert abs(computed["A"] - 0.0) < 1e-9


def test_improvability_ties_zero():
    raw = pd.DataFrame({"d1": [0.9, 0.9], "d2": [0.9, 0.5]}, index=["A", "B"])
    result = compute_improvability(
        raw, {"d1": "max", "d2": "max"}, {"d1": 1.0, "d2": 1.0}
    )
    pd_tab = result.per_dataset.set_index(["model", "dataset"])["improvability"]
    # Tie on d1 (both error 0.1 == best) -> both 0 on that dataset
    assert abs(pd_tab.loc[("A", "d1")]) < 1e-9
    assert abs(pd_tab.loc[("B", "d1")]) < 1e-9


def test_improvability_zero_error_finite():
    raw = pd.DataFrame({"d1": [1.0, 0.5], "d2": [0.8, 0.6]}, index=["A", "B"])
    result = compute_improvability(
        raw, {"d1": "max", "d2": "max"}, {"d1": 1.0, "d2": 1.0}
    )
    # A hits optimum on d1 (error 0) -> improvability 0.0, no NaN/inf
    assert np.isfinite(result.table["improvability"].to_numpy()).all()
    assert np.isfinite(result.per_dataset["improvability"].to_numpy()).all()
    pd_tab = result.per_dataset.set_index(["model", "dataset"])["improvability"]
    assert abs(pd_tab.loc[("A", "d1")]) < 1e-9


def test_improvability_sorted_ascending():
    result = _max_result()
    assert result.table["improvability"].is_monotonic_increasing


def test_improvability_table_schema():
    result = _max_result()
    assert result.table.columns.tolist() == ["model", "improvability"]


def test_improvability_per_dataset_schema():
    result = _max_result()
    assert result.per_dataset.columns.tolist() == [
        "model",
        "dataset",
        "error",
        "best_error",
        "improvability",
    ]


def test_improvability_result_type():
    result = _max_result()
    assert isinstance(result, evaluma.results.ImprovabilityResult)


# --- Slice 3: Benchmark integration + public API ---


def test_improvability_ranking_end_to_end(bench):
    result = bench.improvability_ranking()
    assert isinstance(result, evaluma.results.ImprovabilityResult)
    computed = result.table.set_index("model")["improvability"].to_dict()
    # acc (max, optimum 1.0). errors = 1 - score; per-dataset best over models.
    # A best on every dataset -> 0; B -> 50; C -> 64.34752747...
    assert abs(computed["A"] - 0.0) < 1e-6
    assert abs(computed["B"] - 50.0) < 1e-6
    assert abs(computed["C"] - 64.34752747252747) < 1e-6


def test_improvability_ranking_min_metric(score_df_lowerisbetter):
    bench = evaluma.load_df(
        score_df_lowerisbetter,
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    result = bench.improvability_ranking()
    computed = result.table.set_index("model")["improvability"].to_dict()
    # rmse (min, optimum 0.0): error == score. C is best on every dataset.
    assert abs(computed["C"] - 0.0) < 1e-6
    assert abs(computed["B"] - 44.375) < 1e-6
    assert abs(computed["A"] - 61.55753968253968) < 1e-6


def test_improvability_ranking_seeded_matches_plain(bench_runs, bench):
    runs = bench_runs.improvability_ranking().table.set_index("model")["improvability"]
    plain = bench.improvability_ranking().table.set_index("model")["improvability"]
    for model in ["A", "B", "C"]:
        assert abs(runs.loc[model] - plain.loc[model]) < 1e-9


def test_improvability_ranking_unknown_metric_raises():
    rows = [
        {"model": m, "dataset": d, "metric": "crps", "score": s}
        for m, s in [("A", 0.2), ("B", 0.4)]
        for d in ["d1", "d2"]
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
    with pytest.raises(ValueError, match="crps"):
        bench.improvability_ranking()


def test_improvability_ranking_unknown_metric_min_override_supported():
    rows = [
        {"model": m, "dataset": d, "metric": "crps", "score": s}
        for (m, s_d1, s_d2) in [("A", 0.2, 0.1), ("B", 0.4, 0.3)]
        for d, s in [("d1", s_d1), ("d2", s_d2)]
    ]
    bench = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
        metric_direction={"d1": "min", "d2": "min"},
    )
    result = bench.improvability_ranking()
    computed = result.table.set_index("model")["improvability"].to_dict()
    assert abs(computed["A"] - 0.0) < 1e-9
    assert (
        abs(computed["B"] - ((0.4 - 0.2) / 0.4 * 100 + (0.3 - 0.1) / 0.3 * 100) / 2)
        < 1e-9
    )


def test_improvability_ranking_unknown_metric_max_override_still_raises():
    rows = [
        {"model": m, "dataset": d, "metric": "crps", "score": s}
        for m, s in [("A", 0.2), ("B", 0.4)]
        for d in ["d1", "d2"]
    ]
    bench = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
        metric_direction={"d1": "max", "d2": "max"},
    )
    with pytest.raises(ValueError, match="metric_direction='min'"):
        bench.improvability_ranking()


def test_improvability_ranking_survives_select(bench):
    result = bench.select_models(["A", "B"]).improvability_ranking()
    assert set(result.table["model"]) == {"A", "B"}
    # With C dropped, A is still best on every dataset -> 0.
    computed = result.table.set_index("model")["improvability"].to_dict()
    assert abs(computed["A"] - 0.0) < 1e-6


def test_improvability_ranking_geobench_alias_metric_runs():
    bench = evaluma.load_csv(
        "tests/fixtures/geobench_subset.csv",
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    result = bench.improvability_ranking()
    assert result.table.shape == (5, 2)
    assert result.table["improvability"].between(0.0, 100.0).all()
    assert result.per_dataset["dataset"].nunique() == 5


def test_improvability_result_importable():
    from evaluma import ImprovabilityResult  # noqa: F401


# --- Slice 4: improvability bar chart ---


def test_improvability_plot_returns_figure(bench):
    fig = bench.improvability_ranking().plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_improvability_plot_accepts_ax(bench):
    host_fig, ax = plt.subplots()
    fig = bench.improvability_ranking().plot(ax=ax)
    assert fig is host_fig
    plt.close(fig)


def test_improvability_plot_no_ci(bench):
    result = bench.improvability_ranking()
    n_models = len(result.table)
    fig = result.plot()
    ax = fig.axes[0]
    assert len(ax.patches) == n_models
    plt.close(fig)
