import pandas as pd
import pytest

import evaluma
from evaluma.benchmark import Benchmark


def test_evaluma_importable():
    assert hasattr(evaluma, "__version__")
    assert len(evaluma.__version__) > 0


def test_load_returns_benchmark(score_df):
    b = evaluma.load_df(
        score_df,
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    assert isinstance(b, Benchmark)


def test_benchmark_has_normalized_matrix(bench):
    assert hasattr(bench, "scores_")
    import pandas as pd

    assert isinstance(bench.scores_, pd.DataFrame)
    assert bench.scores_.shape == (3, 4)
    assert (bench.scores_.values >= 0.0).all()
    assert (bench.scores_.values <= 1.0).all()


def test_select_models_returns_new(bench):
    b2 = bench.select_models(["A", "B"])
    assert b2 is not bench


def test_select_models_filters_correctly(bench):
    b2 = bench.select_models(["A", "B"])
    assert b2.scores_.index.tolist() == ["A", "B"]


def test_select_datasets_filters_correctly(bench):
    b2 = bench.select_datasets(["d1", "d2"])
    assert b2.scores_.columns.tolist() == ["d1", "d2"]


def test_drop_incomplete_removes_model(score_df_missing):
    b = evaluma.load_df(
        score_df_missing,
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
        drop_incomplete=True,
    )
    assert b.scores_.shape[0] == 2


def test_drop_incomplete_method_directly(bench):
    b2 = bench.drop_incomplete()
    assert b2.scores_.shape == (3, 4)  # all models complete → nothing dropped


def test_benchmark_immutable(bench):
    bench.select_models(["A"])
    assert bench.scores_.shape == (3, 4)


def test_raw_runs_none_without_seeds(bench):
    assert bench._raw_runs is None


def test_raw_runs_stored_with_seeds(bench_runs):
    assert isinstance(bench_runs._raw_runs, pd.DataFrame)
    assert list(bench_runs._raw_runs.columns) == ["model", "dataset", "seed", "score"]
    assert len(bench_runs._raw_runs) == 36


def test_scores_cached_property_averages_runs(bench_runs, bench):
    assert bench_runs.scores_.shape == (3, 4)
    for model in ["A", "B", "C"]:
        for dataset in ["d1", "d2", "d3", "d4"]:
            assert (
                abs(
                    bench_runs.scores_.loc[model, dataset]
                    - bench.scores_.loc[model, dataset]
                )
                < 1e-9
            )


def test_scores_cached_property_no_seeds_unchanged(bench):
    # regression guard
    assert bench.scores_.shape == (3, 4)
    assert (bench.scores_.values >= 0.0).all()
    assert (bench.scores_.values <= 1.0).all()


def test_select_models_filters_raw_runs(bench_runs):
    b2 = bench_runs.select_models(["A", "B"])
    assert len(b2._raw_runs) == 24


def test_load_missing_cell_hard_error(score_df_missing):
    with pytest.raises(ValueError):
        evaluma.load_df(
            score_df_missing,
            model="model",
            dataset="dataset",
            metric="metric",
            score="score",
            norm_ref_low=0.0,
            norm_ref_high=1.0,
        )
