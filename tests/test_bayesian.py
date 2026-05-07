import baycomp
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import evaluma


def test_bayesian_matches_baycomp_directly(bench):
    vec_a = bench.scores_.loc["A"].values
    vec_b = bench.scores_.loc["B"].values
    ref = baycomp.two_on_multiple(vec_a, vec_b, rope=0.01, random_state=0)
    result = bench.bayesian_comparison(pairs=[("A", "B")], rope=0.01, random_state=0)
    row = result.table[result.table["model_a"] == "A"].iloc[0]
    assert abs(row["p_a_better"] - ref[0]) < 1e-9
    assert abs(row["p_equiv"] - ref[1]) < 1e-9
    assert abs(row["p_b_better"] - ref[2]) < 1e-9


def test_bayesian_all_pairs_row_count(bench):
    result = bench.bayesian_comparison()
    assert len(result.table) == 3


def test_bayesian_reference_mode(bench):
    result = bench.bayesian_comparison(reference="A")
    assert len(result.table) == 2


def test_bayesian_probs_sum_to_one(bench):
    result = bench.bayesian_comparison()
    sums = (
        result.table["p_a_better"]
        + result.table["p_equiv"]
        + result.table["p_b_better"]
    )
    assert np.allclose(sums.values, 1.0, atol=1e-6)


def test_bayesian_default_rope(bench):
    r1 = bench.bayesian_comparison(pairs=[("A", "B")], random_state=0)
    r2 = bench.bayesian_comparison(pairs=[("A", "B")], rope=0.01, random_state=0)
    assert np.allclose(
        r1.table["p_a_better"].values, r2.table["p_a_better"].values, atol=1e-9
    )


def test_bayesian_table_schema(bench):
    result = bench.bayesian_comparison()
    assert result.table.columns.tolist() == [
        "model_a",
        "model_b",
        "p_a_better",
        "p_equiv",
        "p_b_better",
    ]


def test_bayesian_plot_returns_figure(bench):
    fig = bench.bayesian_comparison().plot()
    assert isinstance(fig, Figure)


def test_bayesian_reference_plot_returns_figure(bench):
    result = bench.bayesian_comparison(reference="A")
    assert result.reference == "A"
    fig = result.plot()
    assert isinstance(fig, Figure)


def test_bayesian_symmetry(bench):
    r_ab = bench.bayesian_comparison(pairs=[("A", "B")], random_state=0)
    r_ba = bench.bayesian_comparison(pairs=[("B", "A")], random_state=0)
    row_ab = r_ab.table.iloc[0]
    row_ba = r_ba.table.iloc[0]
    assert abs(row_ab["p_a_better"] - row_ba["p_b_better"]) < 1e-9
    assert abs(row_ab["p_b_better"] - row_ba["p_a_better"]) < 1e-9
    assert abs(row_ab["p_equiv"] - row_ba["p_equiv"]) < 1e-9


def test_bayesian_within_rope_equiv_dominant():
    rope = 0.05
    rows = []
    for m, delta in [("A", 0.0), ("B", 0.01)]:
        for d in ["d1", "d2", "d3", "d4"]:
            rows.append(
                {"model": m, "dataset": d, "metric": "acc", "score": 0.5 + delta}
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
    result = b.bayesian_comparison(pairs=[("A", "B")], rope=rope, random_state=0)
    row = result.table.iloc[0]
    assert row["p_equiv"] > 0.5


def test_bayesian_domination():
    rows = []
    for d in ["d1", "d2", "d3", "d4"]:
        rows.append({"model": "A", "dataset": d, "metric": "acc", "score": 0.95})
        rows.append({"model": "B", "dataset": d, "metric": "acc", "score": 0.05})
    b = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    result = b.bayesian_comparison(pairs=[("A", "B")], rope=0.01, random_state=0)
    row = result.table.iloc[0]
    assert row["p_a_better"] > 0.9


def test_bayesian_tied_models_equiv_near_one():
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
    result = b.bayesian_comparison(pairs=[("X", "Y")], rope=0.01, random_state=0)
    assert result.table.iloc[0]["p_equiv"] > 0.9


def test_bayesian_nearly_tied_equiv_dominant():
    rope = 0.05
    rows = []
    for d in ["d1", "d2", "d3", "d4"]:
        rows.append({"model": "A", "dataset": d, "metric": "acc", "score": 0.50})
        rows.append({"model": "B", "dataset": d, "metric": "acc", "score": 0.52})
    b = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    result = b.bayesian_comparison(pairs=[("A", "B")], rope=rope, random_state=0)
    row = result.table.iloc[0]
    assert row["p_equiv"] > row["p_a_better"]
    assert row["p_equiv"] > row["p_b_better"]
