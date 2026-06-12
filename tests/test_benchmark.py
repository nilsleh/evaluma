import warnings

import pandas as pd
import pytest

import evaluma
from evaluma.benchmark import Benchmark


def _make_bench_unbounded(scores_dict, datasets):
    """Build a data-driven (``None``-bounds) Benchmark from a model→scores dict."""
    rows = [
        {"model": m, "dataset": d, "metric": "acc", "score": s}
        for m, scores in scores_dict.items()
        for d, s in zip(datasets, scores)
    ]
    return evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
    )


# Adversarial fixture: dropping C re-scales A/B under data-driven bounds and
# flips their order. With frozen bounds the subset must equal the parent
# restricted to the kept cells.
_ADVERSARIAL = {
    "A": [0.55, 0.55, 0.10],
    "B": [0.45, 0.45, 0.90],
    "C": [1.00, 1.00, 0.50],
}
_ADVERSARIAL_DATASETS = ["d1", "d2", "d3"]


def test_drop_models_freezes_bounds():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parent = _make_bench_unbounded(_ADVERSARIAL, _ADVERSARIAL_DATASETS)
        dropped = parent.drop_models(["C"])
        pd.testing.assert_frame_equal(
            dropped.scores_, parent.scores_.loc[["A", "B"]]
        )
        order_parent = parent.aggregate_ranking().table["model"].tolist()
        order_dropped = dropped.aggregate_ranking().table["model"].tolist()
        ab_parent = [m for m in order_parent if m in ("A", "B")]
        ab_dropped = [m for m in order_dropped if m in ("A", "B")]
        assert ab_parent == ab_dropped


def test_drop_models_freezes_bounds_min_direction():
    # "min" datasets are negated before normalization; the frozen bounds are
    # pre-inversion, so a subset must not invert twice.
    rows = [
        {"model": m, "dataset": d, "metric": "rmse", "score": s}
        for m, scores in _ADVERSARIAL.items()
        for d, s in zip(_ADVERSARIAL_DATASETS, scores)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parent = evaluma.load_df(
            pd.DataFrame(rows),
            model="model",
            dataset="dataset",
            metric="metric",
            score="score",
            metric_direction={d: "min" for d in _ADVERSARIAL_DATASETS},
        )
        dropped = parent.drop_models(["C"])
        pd.testing.assert_frame_equal(
            dropped.scores_, parent.scores_.loc[["A", "B"]]
        )


def test_select_datasets_freezes_bounds():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parent = _make_bench_unbounded(_ADVERSARIAL, _ADVERSARIAL_DATASETS)
        sub = parent.select_datasets(["d1", "d3"])
        pd.testing.assert_frame_equal(sub.scores_, parent.scores_[["d1", "d3"]])


def test_drop_datasets_freezes_bounds():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parent = _make_bench_unbounded(_ADVERSARIAL, _ADVERSARIAL_DATASETS)
        sub = parent.drop_datasets(["d2"])
        pd.testing.assert_frame_equal(sub.scores_, parent.scores_[["d1", "d3"]])


def test_drop_reference_model_no_longer_raises():
    rows = [
        {"model": m, "dataset": d, "metric": "acc", "score": s}
        for m, scores in {
            "A": [0.8, 0.7, 0.6],
            "B": [0.7, 0.6, 0.5],
            "C": [0.2, 0.3, 0.1],
        }.items()
        for d, s in zip(["d1", "d2", "d3"], scores)
    ]
    parent = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low="C",
        norm_ref_high=1.0,
    )
    dropped = parent.drop_models(["C"])  # must not raise (latent crash fix)
    pd.testing.assert_frame_equal(dropped.scores_, parent.scores_.loc[["A", "B"]])


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
