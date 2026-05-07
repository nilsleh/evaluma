import pytest

import evaluma

FIXTURE = "tests/fixtures/geobench_subset.csv"
EXPECTED_AGGREGATE_ORDER = [
    "Model2",
    "Model1",
    "Model4",
    "Model5",
    "Model3",
]


@pytest.fixture(scope="module")
def geo_bench():
    return evaluma.load_csv(
        FIXTURE,
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )


def test_load_geobench_subset(geo_bench):
    assert geo_bench.scores_.shape == (5, 5)
    assert geo_bench.scores_.notna().all().all()


def test_aggregate_runs(geo_bench):
    result = geo_bench.aggregate_ranking()
    assert result.table.shape == (5, 2)
    assert result.table.columns.tolist() == ["model", "score"]


def test_bayesian_runs(geo_bench):
    result = geo_bench.bayesian_comparison()
    assert len(result.table) == 10
    assert result.table.columns.tolist() == [
        "model_a",
        "model_b",
        "p_a_better",
        "p_equiv",
        "p_b_better",
    ]


def test_profiles_runs(geo_bench):
    result = geo_bench.performance_profiles()
    assert result.table.columns.tolist() == ["tau", "model", "fraction_within_tau"]
    assert set(result.table["model"].unique()) == set(geo_bench.scores_.index.tolist())


def test_aggregate_ranking_order_geobench(geo_bench):
    result = geo_bench.aggregate_ranking()
    assert result.table["model"].tolist() == EXPECTED_AGGREGATE_ORDER
