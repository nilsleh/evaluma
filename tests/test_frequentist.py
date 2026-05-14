import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from matplotlib.figure import Figure

import evaluma
from evaluma.methods.frequentist import _holm_correction, compute_frequentist


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


def _make_scores_matrix(scores_dict, datasets=None):
    if datasets is None:
        datasets = ["d1", "d2", "d3", "d4", "d5", "d6"]
    return pd.DataFrame(scores_dict, index=datasets).T


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------


def test_raises_k_less_than_2():
    m = _make_scores_matrix({"A": [0.5, 0.6, 0.7, 0.8, 0.9, 0.7]})
    with pytest.raises(ValueError, match="at least 2 models"):
        compute_frequentist(m)


def test_raises_n_less_than_5():
    m = _make_scores_matrix(
        {"A": [0.5, 0.6, 0.7, 0.8], "B": [0.4, 0.5, 0.6, 0.7]},
        datasets=["d1", "d2", "d3", "d4"],
    )
    with pytest.raises(ValueError, match="at least 5 datasets"):
        compute_frequentist(m)


def test_raises_bad_reference():
    m = _make_scores_matrix({"A": [0.5, 0.6, 0.7, 0.8, 0.9, 0.7], "B": [0.4, 0.5, 0.6, 0.7, 0.8, 0.6]})
    with pytest.raises(ValueError, match="not found in scores matrix"):
        compute_frequentist(m, reference="Z")


# ---------------------------------------------------------------------------
# Friedman warning
# ---------------------------------------------------------------------------


def test_friedman_warning_when_not_significant():
    # Nearly identical models → Friedman likely not significant
    scores = [0.5, 0.51, 0.49, 0.50, 0.52, 0.50]
    m = _make_scores_matrix({"A": scores, "B": scores})
    with pytest.warns(UserWarning, match="Friedman test not significant"):
        compute_frequentist(m)


def test_no_friedman_warning_when_significant():
    # Clearly different models → Friedman significant → no UserWarning
    b = _make_bench({
        "A": [0.95, 0.92, 0.93, 0.94, 0.96, 0.91],
        "B": [0.10, 0.12, 0.11, 0.13, 0.09, 0.14],
    })
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        b.frequentist_comparison()
    assert all("Friedman" not in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# FrequentistResult fields
# ---------------------------------------------------------------------------


def test_result_has_friedman_fields():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
    })
    res = b.frequentist_comparison()
    assert hasattr(res, "friedman_statistic")
    assert hasattr(res, "friedman_p_value")
    assert isinstance(res.friedman_statistic, float)
    assert isinstance(res.friedman_p_value, float)


def test_all_pairs_cd_not_none():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    res = b.frequentist_comparison()
    assert res.cd is not None
    assert res.cd > 0


def test_reference_cd_is_none():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    res = b.frequentist_comparison(reference="A")
    assert res.cd is None


# ---------------------------------------------------------------------------
# All-pairs schema: rank_diff present, no p_value_corrected
# ---------------------------------------------------------------------------


def test_all_pairs_column_names():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison()
    assert list(result.table.columns) == [
        "model_a",
        "model_b",
        "rank_diff",
        "p_value",
        "significant",
    ]


def test_all_pairs_no_p_value_corrected():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison()
    assert "p_value_corrected" not in result.table.columns


def test_all_pairs_row_count():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison()
    assert len(result.table) == 3  # C(3, 2) = 3


def test_all_pairs_rank_diff_is_positive():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison()
    assert (result.table["rank_diff"] >= 0).all()


def test_all_pairs_significant_consistent_with_alpha():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison(alpha=0.05)
    expected = result.table["p_value"] < 0.05
    assert (result.table["significant"] == expected.values).all()


# ---------------------------------------------------------------------------
# Reference schema: w_statistic present, p_value_corrected present
# ---------------------------------------------------------------------------


def test_reference_column_names():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison(reference="A")
    assert list(result.table.columns) == [
        "model_a",
        "model_b",
        "w_statistic",
        "p_value",
        "p_value_corrected",
        "significant",
    ]


def test_reference_row_count():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison(reference="A")
    assert len(result.table) == 2  # k - 1


def test_reference_model_a_is_reference():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison(reference="A")
    assert (result.table["model_a"] == "A").all()


def test_reference_significant_uses_corrected_p():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison(reference="A", alpha=0.05)
    expected = result.table["p_value_corrected"] < 0.05
    assert (result.table["significant"] == expected.values).all()


def test_identical_models_p_value_one():
    scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.7]
    b = _make_bench({"A": scores, "B": scores, "C": [0.3, 0.4, 0.35, 0.45, 0.38, 0.42]})
    result = b.frequentist_comparison(reference="A")
    ab_row = result.table[result.table["model_b"] == "B"]
    assert ab_row.iloc[0]["p_value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CD scalar consistency: rank_diff > cd ↔ significant (all-pairs mode)
# ---------------------------------------------------------------------------


def test_cd_scalar_consistency():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison(alpha=0.05)
    cd = result.cd
    for _, row in result.table.iterrows():
        # rank_diff > cd ↔ significant (may differ only at floating-point boundary)
        rank_sig = row["rank_diff"] > cd
        p_sig = row["significant"]
        # They should agree to within floating-point tolerance
        # (both derived from the same studentized_range distribution at df=inf)
        assert rank_sig == p_sig or abs(row["rank_diff"] - cd) < 1e-10


# ---------------------------------------------------------------------------
# avg_ranks attribute
# ---------------------------------------------------------------------------


def test_avg_ranks_stored():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison()
    assert hasattr(result, "avg_ranks")
    assert set(result.avg_ranks.index) == {"A", "B", "C"}


def test_avg_ranks_ordering():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison()
    # A has highest scores → lowest (best) rank
    assert result.avg_ranks["A"] < result.avg_ranks["B"] < result.avg_ranks["C"]


# ---------------------------------------------------------------------------
# Clique detection
# ---------------------------------------------------------------------------


def test_clique_detection_identical_models():
    # Two nearly identical models should be in the same clique
    b = _make_bench({
        "A": [0.90, 0.88, 0.91, 0.89, 0.87, 0.92],
        "B": [0.89, 0.87, 0.90, 0.88, 0.86, 0.91],
        "C": [0.10, 0.12, 0.11, 0.13, 0.09, 0.14],
    })
    result = b.frequentist_comparison(alpha=0.05)
    # A and B should NOT be significantly different (rank_diff ≤ cd)
    ab_row = result.table[
        ((result.table["model_a"] == "A") & (result.table["model_b"] == "B"))
        | ((result.table["model_a"] == "B") & (result.table["model_b"] == "A"))
    ]
    if not ab_row.empty:
        assert not ab_row.iloc[0]["significant"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_all_pairs_returns_figure():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison()
    fig = result.plot()
    assert isinstance(fig, Figure)


def test_plot_reference_returns_figure():
    b = _make_bench({
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28],
    })
    result = b.frequentist_comparison(reference="A")
    fig = result.plot()
    assert isinstance(fig, Figure)


def test_plot_with_clique_bar():
    b = _make_bench({
        "A": [0.90, 0.88, 0.91, 0.89, 0.87, 0.92],
        "B": [0.89, 0.87, 0.90, 0.88, 0.86, 0.91],
        "C": [0.10, 0.12, 0.11, 0.13, 0.09, 0.14],
    })
    result = b.frequentist_comparison(alpha=0.05)
    fig = result.plot()
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# Holm correction (reused in reference mode)
# ---------------------------------------------------------------------------


def test_holm_matches_statsmodels():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.multitest import multipletests

    raw_p = [0.01, 0.04, 0.20, 0.15, 0.03]
    result = _holm_correction(raw_p)
    _, corrected, _, _ = multipletests(raw_p, method="holm")
    np.testing.assert_allclose(result, corrected, atol=1e-12)


def test_holm_single_p_value():
    result = _holm_correction([0.03])
    np.testing.assert_allclose(result, [0.03])


def test_holm_empty():
    assert len(_holm_correction([])) == 0


def test_holm_clamps_to_one():
    raw_p = [0.9, 0.85, 0.95]
    result = _holm_correction(raw_p)
    assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# CLI subcommand
# ---------------------------------------------------------------------------


def _make_csv(tmp_path, n_datasets=6):
    datasets = [f"d{i}" for i in range(1, n_datasets + 1)]
    scores = {
        "A": [0.9, 0.8, 0.7, 0.6, 0.85, 0.75][:n_datasets],
        "B": [0.5, 0.6, 0.4, 0.5, 0.55, 0.45][:n_datasets],
        "C": [0.2, 0.3, 0.25, 0.35, 0.22, 0.28][:n_datasets],
    }
    rows = [
        {"model": m, "dataset": d, "metric": "acc", "score": s}
        for m, sc in scores.items()
        for d, s in zip(datasets, sc)
    ]
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_frequentist_subcommand_creates_files(tmp_path):
    from evaluma.cli import main

    csv_path = _make_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "frequentist",
            str(csv_path),
            "--model", "model",
            "--dataset", "dataset",
            "--metric", "metric",
            "--score", "score",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "frequentist_comparison.csv").exists()
    assert (out / "frequentist_comparison.png").exists()


def test_frequentist_subcommand_reference_flag(tmp_path):
    from evaluma.cli import main

    csv_path = _make_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "frequentist",
            str(csv_path),
            "--model", "model",
            "--dataset", "dataset",
            "--metric", "metric",
            "--score", "score",
            "--reference", "A",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out / "frequentist_comparison.csv")
    assert len(df) == 2  # k - 1 rows
    assert "w_statistic" in df.columns
    assert "p_value_corrected" in df.columns
