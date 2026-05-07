"""Tests for evaluma.load_df() with the metric_type_bounds parameter."""

import pandas as pd
import pytest

import evaluma


def _make_df(rows):
    return pd.DataFrame(rows)


def _iou_rows():
    """Three models, two IoU datasets."""
    return [
        {"model": m, "dataset": d, "metric": "iou", "score": s}
        for m, scores in [
            ("A", [0.9, 0.8]),
            ("B", [0.6, 0.7]),
            ("C", [0.3, 0.4]),
        ]
        for d, s in zip(["d_iou_0", "d_iou_1"], scores)
    ]


def _rmse_rows():
    """Three models, two RMSE datasets."""
    return [
        {"model": m, "dataset": d, "metric": "rmse", "score": s}
        for m, scores in [
            ("A", [1.5, 2.0]),
            ("B", [2.5, 3.0]),
            ("baseline", [4.0, 5.0]),
        ]
        for d, s in zip(["d_rmse_0", "d_rmse_1"], scores)
    ]


def _mixed_rows():
    """Models with both IoU and RMSE datasets (baseline present for RMSE ceiling)."""
    iou = [
        {"model": m, "dataset": d, "metric": "iou", "score": s}
        for m, scores in [
            ("A", [0.9, 0.8]),
            ("B", [0.6, 0.7]),
            ("baseline", [0.4, 0.5]),
        ]
        for d, s in zip(["d_iou_0", "d_iou_1"], scores)
    ]
    rmse = [
        {"model": m, "dataset": d, "metric": "rmse", "score": s}
        for m, scores in [
            ("A", [1.5, 2.0]),
            ("B", [2.5, 3.0]),
            ("baseline", [4.0, 5.0]),
        ]
        for d, s in zip(["d_rmse_0", "d_rmse_1"], scores)
    ]
    return iou + rmse


# ---------------------------------------------------------------------------
# Bounded metrics without explicit metric_type_bounds entries
# ---------------------------------------------------------------------------

def test_bounded_metrics_no_bounds_spec():
    """IoU datasets with metric_type_bounds={} silently use natural [0,1] bounds."""
    df = _make_df(_iou_rows())
    bench = evaluma.load_df(df, metric_type_bounds={})
    norm = bench.scores_
    # All IoU scores are already in [0,1], so normalized values must be in [0,1]
    assert (norm >= 0).all().all()
    assert (norm <= 1).all().all()


# ---------------------------------------------------------------------------
# Regression metric with scalar ceiling
# ---------------------------------------------------------------------------

def test_regression_metric_with_scalar_high():
    """RMSE dataset with explicit scalar ceiling produces normalized scores in [0,1]."""
    df = _make_df(_rmse_rows())
    bench = evaluma.load_df(df, metric_type_bounds={"rmse": (0.0, 5.0)})
    norm = bench.scores_
    # Model A has RMSE 1.5 and 2.0 → after direction flip, normalized > model B
    assert (norm >= 0).all().all()
    assert (norm <= 1).all().all()
    # Lower RMSE → higher normalized score
    assert norm.loc["A", "d_rmse_0"] > norm.loc["B", "d_rmse_0"]


# ---------------------------------------------------------------------------
# Regression metric with model-name ceiling
# ---------------------------------------------------------------------------

def test_regression_metric_with_model_name_high():
    """metric_type_bounds with a model name uses that model's per-dataset score as ceiling."""
    df = _make_df(_rmse_rows())
    bench = evaluma.load_df(df, metric_type_bounds={"rmse": (0.0, "baseline")})
    norm = bench.scores_
    # baseline has RMSE == ceiling → normalized score = 0
    assert norm.loc["baseline", "d_rmse_0"] == pytest.approx(0.0)
    assert norm.loc["baseline", "d_rmse_1"] == pytest.approx(0.0)
    # Model A has lowest RMSE → highest normalized score
    assert norm.loc["A", "d_rmse_0"] > norm.loc["B", "d_rmse_0"]


# ---------------------------------------------------------------------------
# Mixed IoU + RMSE benchmark
# ---------------------------------------------------------------------------

def test_mixed_iou_and_rmse():
    """Benchmark with IoU and RMSE datasets; all normalized scores in [0, 1]."""
    df = _make_df(_mixed_rows())
    bench = evaluma.load_df(
        df,
        metric_type_bounds={"rmse": (0.0, "baseline")},
    )
    norm = bench.scores_
    assert (norm >= 0).all().all()
    assert (norm <= 1).all().all()
    # IoU higher is better; RMSE lower raw score → higher normalized score
    assert norm.loc["A", "d_iou_0"] > norm.loc["B", "d_iou_0"]
    assert norm.loc["A", "d_rmse_0"] > norm.loc["B", "d_rmse_0"]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_conflict_raises():
    """Combining metric_type_bounds with norm_ref_high raises ValueError."""
    df = _make_df(_iou_rows())
    with pytest.raises(ValueError, match="metric_type_bounds cannot be combined"):
        evaluma.load_df(df, metric_type_bounds={"iou": (0.0, 1.0)}, norm_ref_high=1.0)


def test_conflict_with_norm_ref_low_raises():
    df = _make_df(_iou_rows())
    with pytest.raises(ValueError, match="metric_type_bounds cannot be combined"):
        evaluma.load_df(df, metric_type_bounds={"iou": (0.0, 1.0)}, norm_ref_low=0.0)


def test_regression_metric_no_bounds_raises():
    """RMSE dataset with no metric_type_bounds entry raises ValueError."""
    df = _make_df(_rmse_rows())
    with pytest.raises(ValueError, match="no natural upper bound"):
        evaluma.load_df(df, metric_type_bounds={})


def test_unknown_metric_raises():
    """Dataset with an unregistered metric name raises ValueError."""
    rows = [
        {"model": m, "dataset": "d1", "metric": "crps", "score": s}
        for m, s in [("A", 0.5), ("B", 0.7), ("C", 0.6)]
    ]
    df = _make_df(rows)
    with pytest.raises(ValueError, match="Unknown metric 'crps'"):
        evaluma.load_df(df, metric_type_bounds={})


def test_metric_direction_override():
    """Explicit metric_direction overrides the registry direction."""
    # IoU is "max" by default; override one dataset to "min" and verify the
    # normalized ranking flips.
    df = _make_df(_iou_rows())
    bench_normal = evaluma.load_df(df, metric_type_bounds={})
    bench_flipped = evaluma.load_df(
        df,
        metric_type_bounds={},
        metric_direction={"d_iou_0": "min"},
    )
    # In normal mode A > B on d_iou_0; after flip the ranking should invert
    assert bench_normal.scores_.loc["A", "d_iou_0"] > bench_normal.scores_.loc["B", "d_iou_0"]
    assert bench_flipped.scores_.loc["A", "d_iou_0"] < bench_flipped.scores_.loc["B", "d_iou_0"]


def test_model_name_not_found_raises():
    """Using a nonexistent model name as ceiling raises ValueError."""
    df = _make_df(_rmse_rows())
    with pytest.raises(ValueError, match="not found in score matrix"):
        evaluma.load_df(df, metric_type_bounds={"rmse": (0.0, "nonexistent")})
