import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import evaluma
from evaluma.methods.elo import (
    _fit_elo,
    compute_battles,
    compute_battles_from_runs,
    compute_elo,
    compute_winrate_matrix,
)
from evaluma.results import EloResult

matplotlib.use("Agg")


def _scores(data: dict, datasets=None) -> pd.DataFrame:
    """model×dataset score DataFrame (normalized, higher=better)."""
    if datasets is None:
        datasets = [f"d{i}" for i in range(len(next(iter(data.values()))))]
    return pd.DataFrame(data, index=datasets).T


# ---------------------------------------------------------------------------
# Slice 1: Battle generation
# ---------------------------------------------------------------------------


def test_battles_count():
    scores = _scores(
        {
            "A": [0.9, 0.8, 0.7, 0.6],
            "B": [0.5, 0.6, 0.4, 0.5],
            "C": [0.2, 0.3, 0.25, 0.35],
        }
    )
    battles = compute_battles(scores)
    M, N = 3, 4
    assert len(battles) == M * (M - 1) // 2 * N


def test_battles_columns():
    scores = _scores({"A": [0.9, 0.8], "B": [0.5, 0.6]})
    battles = compute_battles(scores)
    assert battles.columns.tolist() == [
        "model_a",
        "model_b",
        "outcome",
        "dataset",
        "weight",
    ]


def test_battles_outcome_binary():
    scores = _scores(
        {
            "A": [0.9, 0.8, 0.7],
            "B": [0.5, 0.6, 0.4],
            "C": [0.2, 0.3, 0.25],
        }
    )
    battles = compute_battles(scores, tie_threshold=None)
    assert set(battles["outcome"].unique()).issubset({0.0, 1.0})


def test_battles_weight_per_dataset():
    scores = _scores(
        {
            "A": [0.9, 0.8, 0.7],
            "B": [0.5, 0.6, 0.4],
            "C": [0.2, 0.3, 0.25],
        }
    )
    battles = compute_battles(scores)
    for d in scores.columns.tolist():
        total = battles[battles["dataset"] == d]["weight"].sum()
        assert abs(total - 1.0) < 1e-9, f"dataset {d}: weights sum to {total}"


def test_battles_correct_winner():
    scores = _scores({"A": [0.9], "B": [0.5]})
    battles = compute_battles(scores)
    assert len(battles) == 1
    row = battles.iloc[0]
    assert row["model_a"] == "A"
    assert row["model_b"] == "B"
    assert row["outcome"] == 1.0


def test_battles_tie_threshold_emits_tie():
    # Diff 0.02 < threshold 0.05 → tie battle with outcome=0.5
    scores = _scores({"A": [0.52], "B": [0.50]})
    battles = compute_battles(scores, tie_threshold=0.05)
    assert len(battles) == 1
    assert battles.iloc[0]["outcome"] == 0.5


def test_battles_equal_scores_emits_tie():
    # Exact equality → tie battle with outcome=0.5
    scores = _scores({"A": [0.5], "B": [0.5]})
    battles = compute_battles(scores, tie_threshold=None)
    assert len(battles) == 1
    assert battles.iloc[0]["outcome"] == 0.5


def test_battles_small_diff_above_threshold_kept():
    # Diff 0.06 > threshold 0.05 → battle is kept
    scores = _scores({"A": [0.56], "B": [0.50]})
    battles = compute_battles(scores, tie_threshold=0.05)
    assert len(battles) == 1
    assert battles.iloc[0]["outcome"] == 1.0


# ---------------------------------------------------------------------------
# Slice 2: Win-rate matrix
# ---------------------------------------------------------------------------


def test_winrate_matrix_shape():
    scores = _scores(
        {
            "A": [0.9, 0.8, 0.7],
            "B": [0.5, 0.6, 0.4],
            "C": [0.2, 0.3, 0.25],
        }
    )
    wm = compute_winrate_matrix(scores)
    assert wm.shape == (3, 3)


def test_winrate_diagonal_nan():
    scores = _scores(
        {
            "A": [0.9, 0.8],
            "B": [0.5, 0.6],
            "C": [0.2, 0.3],
        }
    )
    wm = compute_winrate_matrix(scores)
    for model in wm.index:
        assert pd.isna(wm.loc[model, model])


def test_winrate_symmetry():
    scores = _scores(
        {
            "A": [0.9, 0.8, 0.7],
            "B": [0.5, 0.6, 0.4],
            "C": [0.2, 0.3, 0.25],
        }
    )
    wm = compute_winrate_matrix(scores)
    models = wm.index.tolist()
    for i in models:
        for j in models:
            if i != j:
                assert wm.loc[i, j] + wm.loc[j, i] == pytest.approx(1.0)


def test_winrate_dominant_model():
    scores = _scores(
        {
            "A": [0.9, 0.9, 0.9],
            "B": [0.5, 0.5, 0.5],
            "C": [0.2, 0.2, 0.2],
        }
    )
    wm = compute_winrate_matrix(scores)
    assert wm.loc["A"].dropna().eq(1.0).all()


def test_winrate_sorted_by_avg():
    scores = _scores(
        {
            "A": [0.9, 0.8, 0.7],
            "B": [0.5, 0.6, 0.4],
            "C": [0.2, 0.3, 0.25],
        }
    )
    wm = compute_winrate_matrix(scores)
    avg = wm.mean(axis=1, skipna=True)
    assert avg.is_monotonic_decreasing


def test_winrate_exact_value():
    # A beats B on d0 and d2, B beats A on d1 → win-rate A vs B = 2/3
    scores = _scores(
        {
            "A": [0.9, 0.4, 0.9],
            "B": [0.5, 0.8, 0.5],
        }
    )
    wm = compute_winrate_matrix(scores)
    assert wm.loc["A", "B"] == pytest.approx(2 / 3)
    assert wm.loc["B", "A"] == pytest.approx(1 / 3)


def test_winrate_index_equals_columns():
    scores = _scores(
        {
            "A": [0.9, 0.8],
            "B": [0.5, 0.6],
            "C": [0.2, 0.3],
        }
    )
    wm = compute_winrate_matrix(scores)
    assert wm.index.tolist() == wm.columns.tolist()


def test_winrate_tie_threshold_gives_half():
    # A-B diff 0.02 < threshold 0.05 → tie → win-rate = 0.5 for both
    scores = _scores({"A": [0.52, 0.52], "B": [0.50, 0.50]})
    wm = compute_winrate_matrix(scores, tie_threshold=0.05)
    assert wm.loc["A", "B"] == pytest.approx(0.5)
    assert wm.loc["B", "A"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Slice 3: MLE ELO core
# ---------------------------------------------------------------------------


def test_fit_elo_correct_ordering():
    scores = _scores(
        {
            "A": [0.9, 0.9, 0.9, 0.9, 0.9],
            "B": [0.5, 0.5, 0.5, 0.5, 0.5],
            "C": [0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )
    battles = compute_battles(scores)
    ratings = _fit_elo(battles)
    assert ratings["A"] > ratings["B"] > ratings["C"]


def test_fit_elo_standard_scale():
    # A beats B at empirical rate 0.75 over many battles. On the standard ELO
    # scale the gap must equal 400*log10(p/(1-p)) ≈ 190.85, matching TabArena.
    # Guards against the logit→ELO conversion being off by a factor of ln(base).
    n = 4000
    rng = np.random.default_rng(0)
    wins = rng.random(n) < 0.75
    battles = pd.DataFrame(
        [("A", "B", 1.0 if w else 0.0, 0, 1.0) for w in wins],
        columns=["model_a", "model_b", "outcome", "dataset", "weight"],
    )
    ratings = _fit_elo(battles, models=["A", "B"])
    expected = 400 * np.log10(0.75 / 0.25)
    assert (ratings["A"] - ratings["B"]) == pytest.approx(expected, rel=0.02)


def test_fit_elo_symmetric_models():
    scores = _scores(
        {
            "A": [0.5, 0.5, 0.5, 0.5],
            "B": [0.5, 0.5, 0.5, 0.5],
            "C": [0.5, 0.5, 0.5, 0.5],
        }
    )
    battles = compute_battles(scores)
    # All equal → no battles; must pass models explicitly when battles is empty
    ratings = _fit_elo(battles, models=scores.index.tolist())
    assert abs(ratings["A"] - ratings["B"]) < 1e-6
    assert abs(ratings["B"] - ratings["C"]) < 1e-6


def test_fit_elo_returns_series():
    scores = _scores({"A": [0.9, 0.8], "B": [0.5, 0.4]})
    battles = compute_battles(scores)
    ratings = _fit_elo(battles)
    assert isinstance(ratings, pd.Series)


def test_fit_elo_all_models_present():
    scores = _scores(
        {
            "A": [0.9, 0.8, 0.7],
            "B": [0.5, 0.6, 0.4],
            "C": [0.2, 0.3, 0.25],
        }
    )
    battles = compute_battles(scores)
    ratings = _fit_elo(battles)
    assert set(ratings.index) == {"A", "B", "C"}


def test_fit_elo_total_dominance_fallback():
    # A wins every single battle — logistic regression degenerates, fallback fires
    scores = _scores(
        {
            "A": [1.0, 1.0, 1.0, 1.0, 1.0],
            "B": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    battles = compute_battles(scores)
    ratings = _fit_elo(battles)
    assert isinstance(ratings, pd.Series)
    assert ratings.notna().all()
    assert ratings["A"] > ratings["B"]


# ---------------------------------------------------------------------------
# Slice 4: Bootstrap CIs + calibration + compute_elo()
# ---------------------------------------------------------------------------


def _bench_scores():
    return _scores(
        {
            "A": [0.9, 0.8, 0.7, 0.85, 0.95],
            "B": [0.5, 0.6, 0.4, 0.55, 0.45],
            "C": [0.2, 0.3, 0.25, 0.15, 0.35],
        }
    )


def test_compute_elo_table_schema():
    table, _ = compute_elo(_bench_scores(), n_bootstrap=0)
    assert table.columns.tolist() == ["model", "ELO", "CI_low", "CI_high"]


def test_compute_elo_sorted_descending():
    table, _ = compute_elo(_bench_scores(), n_bootstrap=0)
    assert table["ELO"].is_monotonic_decreasing


def test_compute_elo_no_bootstrap():
    table, _ = compute_elo(_bench_scores(), n_bootstrap=0)
    assert table["CI_low"].isna().all()
    assert table["CI_high"].isna().all()


def test_compute_elo_ci_bracketing():
    table, _ = compute_elo(_bench_scores(), n_bootstrap=200, random_state=0)
    assert (table["CI_low"] <= table["ELO"]).all()
    assert (table["ELO"] <= table["CI_high"]).all()


def test_compute_elo_calibration():
    table, _ = compute_elo(_bench_scores(), n_bootstrap=0, calibration_model="A")
    elo_a = table.set_index("model").loc["A", "ELO"]
    assert elo_a == pytest.approx(1000.0)


def test_compute_elo_calibration_unknown_model():
    with pytest.raises(ValueError, match="calibration_model"):
        compute_elo(_bench_scores(), n_bootstrap=0, calibration_model="Z")


def test_compute_elo_reproducible():
    t1, _ = compute_elo(_bench_scores(), n_bootstrap=100, random_state=7)
    t2, _ = compute_elo(_bench_scores(), n_bootstrap=100, random_state=7)
    assert np.allclose(t1["CI_low"].values, t2["CI_low"].values)
    assert np.allclose(t1["CI_high"].values, t2["CI_high"].values)


def test_compute_elo_returns_winrate_matrix():
    table, wm = compute_elo(_bench_scores(), n_bootstrap=0)
    assert isinstance(wm, pd.DataFrame)
    assert wm.shape == (3, 3)
    for m in wm.index:
        assert pd.isna(wm.loc[m, m])


def test_compute_elo_all_models_in_table():
    table, _ = compute_elo(_bench_scores(), n_bootstrap=0)
    assert set(table["model"]) == {"A", "B", "C"}


def test_compute_elo_ci_nonzero_width():
    # Bootstrap should produce some variance for a non-trivial benchmark
    table, _ = compute_elo(_bench_scores(), n_bootstrap=200, random_state=0)
    widths = table["CI_high"] - table["CI_low"]
    assert (widths > 0).all()


def test_compute_elo_calibration_with_bootstrap():
    # With bootstrap, calibration_model's point estimate is still exactly 1000
    table, _ = compute_elo(
        _bench_scores(), n_bootstrap=50, random_state=3, calibration_model="B"
    )
    elo_b = table.set_index("model").loc["B", "ELO"]
    assert elo_b == pytest.approx(1000.0)


def test_compute_elo_tie_threshold_accepted():
    # Passing tie_threshold does not raise and still produces a valid table
    table, _ = compute_elo(_bench_scores(), n_bootstrap=0, tie_threshold=0.1)
    assert table.columns.tolist() == ["model", "ELO", "CI_low", "CI_high"]
    assert len(table) == 3


# ---------------------------------------------------------------------------
# Slice 5: EloResult, Benchmark.elo_ranking(), plots
# ---------------------------------------------------------------------------


def test_elo_ranking_returns_elo_result(bench):
    result = bench.elo_ranking(n_bootstrap=0)
    assert isinstance(result, EloResult)


def test_elo_result_table_schema(bench):
    result = bench.elo_ranking(n_bootstrap=0)
    assert result.table.columns.tolist() == ["model", "ELO", "CI_low", "CI_high"]


def test_elo_result_has_winrate_matrix(bench):
    result = bench.elo_ranking(n_bootstrap=0)
    assert isinstance(result.winrate_matrix, pd.DataFrame)
    assert result.winrate_matrix.shape == (3, 3)


def test_elo_result_winrate_diagonal_nan(bench):
    result = bench.elo_ranking(n_bootstrap=0)
    for m in result.winrate_matrix.index:
        assert pd.isna(result.winrate_matrix.loc[m, m])


def test_elo_plot_returns_figure(bench):
    fig = bench.elo_ranking(n_bootstrap=0).plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_elo_plot_winrate_returns_figure(bench):
    fig = bench.elo_ranking(n_bootstrap=0).plot_winrate()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_elo_result_exported():
    assert hasattr(evaluma, "EloResult")


def test_elo_ranking_passes_params(bench):
    result = bench.elo_ranking(
        n_bootstrap=50, random_state=42, tie_threshold=0.05, calibration_model="A"
    )
    elo_a = result.table.set_index("model").loc["A", "ELO"]
    assert elo_a == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Slice 6: Tie outcome handling
# ---------------------------------------------------------------------------


def test_battles_tie_outcome_is_half():
    scores = _scores({"A": [0.5], "B": [0.5]})
    battles = compute_battles(scores)
    assert battles.iloc[0]["outcome"] == pytest.approx(0.5)


def test_winrate_matrix_tie_counts_half():
    # Two models that always tie → win-rate = 0.5 off-diagonal
    scores = _scores({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
    wm = compute_winrate_matrix(scores)
    assert wm.loc["A", "B"] == pytest.approx(0.5)
    assert wm.loc["B", "A"] == pytest.approx(0.5)


def test_fit_elo_handles_tie_outcome():
    # A always beats C; A ties B; B always beats C → A >= B > C
    battles = pd.DataFrame(
        [
            ("A", "B", 0.5, "d1", 1.0),
            ("A", "C", 1.0, "d1", 1.0),
            ("B", "C", 1.0, "d1", 1.0),
        ],
        columns=["model_a", "model_b", "outcome", "dataset", "weight"],
    )
    ratings = _fit_elo(battles)
    assert ratings["A"] >= ratings["B"]
    assert ratings["B"] > ratings["C"]
    assert ratings.notna().all()


# ---------------------------------------------------------------------------
# Slice 7: compute_battles_from_runs
# ---------------------------------------------------------------------------


def _make_raw_runs(models, datasets, seeds, scores_fn):
    """Build a long-format raw_runs DataFrame."""
    rows = []
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "score": scores_fn(model, dataset, seed),
                    }
                )
    return pd.DataFrame(rows)


def test_compute_battles_from_runs_basic():
    # 3 models × 2 datasets × 2 seeds → 3 pairs × 2 seeds × 2 datasets = 12 battles
    raw_runs = _make_raw_runs(
        ["A", "B", "C"],
        ["d1", "d2"],
        [0, 1],
        lambda m, d, s: {"A": 0.9, "B": 0.6, "C": 0.3}[m],
    )
    battles = compute_battles_from_runs(raw_runs)
    assert len(battles) == 3 * 2 * 2
    assert battles.columns.tolist() == [
        "model_a",
        "model_b",
        "outcome",
        "dataset",
        "seed",
        "weight",
    ]


def test_compute_battles_from_runs_weight_sum():
    # Each dataset should contribute total weight = 1.0
    raw_runs = _make_raw_runs(
        ["A", "B", "C"],
        ["d1", "d2"],
        [0, 1, 2],
        lambda m, d, s: {"A": 0.9, "B": 0.6, "C": 0.3}[m],
    )
    battles = compute_battles_from_runs(raw_runs)
    for dataset in ["d1", "d2"]:
        total = battles[battles["dataset"] == dataset]["weight"].sum()
        assert abs(total - 1.0) < 1e-9, f"dataset {dataset}: weight sum = {total}"


def test_compute_battles_from_runs_metric_direction():
    # With direction "min", lower score = better; A has lower score → should win
    raw_runs = _make_raw_runs(
        ["A", "B"],
        ["d1"],
        [0],
        lambda m, d, s: 0.1 if m == "A" else 0.9,
    )
    battles = compute_battles_from_runs(raw_runs, metric_direction={"d1": "min"})
    assert len(battles) == 1
    row = battles.iloc[0]
    assert row["model_a"] == "A"
    assert row["outcome"] == pytest.approx(1.0)


def test_compute_battles_from_runs_tie():
    raw_runs = _make_raw_runs(
        ["A", "B"],
        ["d1"],
        [0],
        lambda m, d, s: 0.5,
    )
    battles = compute_battles_from_runs(raw_runs)
    assert battles.iloc[0]["outcome"] == pytest.approx(0.5)


def test_compute_elo_with_raw_runs():
    # raw_runs path should give same ranking direction as scores-only on consistent data
    raw_runs = _make_raw_runs(
        ["A", "B", "C"],
        ["d1", "d2", "d3"],
        [0, 1],
        lambda m, d, s: {"A": 0.9, "B": 0.6, "C": 0.3}[m] + s * 0.01,
    )
    scores_data = raw_runs.groupby(["model", "dataset"])["score"].mean().unstack()
    table, wm = compute_elo(scores_data, n_bootstrap=0, raw_runs=raw_runs)
    ranked = table.set_index("model")["ELO"]
    assert ranked["A"] > ranked["B"] > ranked["C"]
    assert isinstance(wm, pd.DataFrame)
