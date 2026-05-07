import pytest

from evaluma.metric_registry import KNOWN_METRICS, get_direction, get_natural_bounds


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
        "r2",
    ):
        assert get_direction(name) == "max"
        low, high = get_natural_bounds(name)
        assert low == 0.0
        assert high == 1.0


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
