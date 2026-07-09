import pytest

from evaluma.metric_registry import (
    KNOWN_METRICS,
    get_direction,
    get_error_optimum,
    get_natural_bounds,
)


def test_known_max_metrics():
    for name in (
        "accuracy",
        "iou",
        "f1",
        "auc",
        "ap",
        "map",
        "precision",
        "recall",
    ):
        assert get_direction(name) == "max"
        low, high = get_natural_bounds(name)
        assert low == 0.0
        assert high == 1.0


def test_r2_bounds():
    # R² ∈ (−∞, 1]: no natural lower bound, so users must supply one.
    assert get_direction("r2") == "max"
    low, high = get_natural_bounds("r2")
    assert low == 0.0
    assert high is None


def test_known_min_metrics():
    for name in ("rmse", "mae", "mse"):
        assert get_direction(name) == "min"
        low, high = get_natural_bounds(name)
        assert low == 0.0
        assert high is None


def test_case_insensitive():
    assert get_direction("RMSE") == "min"
    assert get_direction("Accuracy") == "max"
    assert get_natural_bounds("RMSE") == (0.0, None)


def test_geobench_metric_aliases():
    for name in (
        "Multiclass_Jaccard_Index",
        "Multilabel_F1_Score",
        "Overall_Accuracy",
        "test_test_map",
        "test_test_segm_map",
    ):
        assert get_direction(name) == "max"
        assert get_natural_bounds(name) == (0.0, 1.0)
        assert get_error_optimum(name) == 1.0


def test_unknown_metric_raises_get_direction():
    with pytest.raises(ValueError, match="Unknown metric 'crps'"):
        get_direction("crps")


def test_unknown_metric_raises_get_natural_bounds():
    with pytest.raises(ValueError, match="Unknown metric 'crps'"):
        get_natural_bounds("crps")


def test_known_metrics_all_have_valid_direction():
    for name in KNOWN_METRICS:
        assert get_direction(name) in ("min", "max")


def test_known_metrics_all_have_float_low():
    for name in KNOWN_METRICS:
        low, _ = get_natural_bounds(name)
        assert isinstance(low, float)


def test_get_error_optimum_bounded_max_metrics():
    for name in (
        "accuracy",
        "acc",
        "iou",
        "miou",
        "f1",
        "auc",
        "ap",
        "map",
        "precision",
        "recall",
        "r2",
    ):
        opt = get_error_optimum(name)
        assert isinstance(opt, float)
        assert opt == 1.0


def test_get_error_optimum_min_metrics():
    for name in ("rmse", "mae", "mse", "mape", "logloss", "log_loss"):
        opt = get_error_optimum(name)
        assert isinstance(opt, float)
        assert opt == 0.0


def test_get_error_optimum_case_insensitive():
    assert get_error_optimum("RMSE") == 0.0
    assert get_error_optimum("Accuracy") == 1.0


def test_logloss_registered():
    assert get_direction("logloss") == "min"
    assert get_direction("log_loss") == "min"
    assert get_natural_bounds("logloss") == (0.0, None)
    assert get_natural_bounds("log_loss") == (0.0, None)


def test_get_error_optimum_unknown_raises():
    with pytest.raises(ValueError, match="crps"):
        get_error_optimum("crps")


def test_known_metrics_all_have_error_optimum():
    for name in KNOWN_METRICS:
        assert isinstance(get_error_optimum(name), float)
