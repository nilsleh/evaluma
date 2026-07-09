"""Metric name registry: maps names to optimization direction and natural bounds."""

KNOWN_METRICS = {
    # name: (direction, (natural_low, natural_high))  — None high: user must specify
    "accuracy": ("max", (0.0, 1.0)),
    "acc": ("max", (0.0, 1.0)),
    "overall_accuracy": ("max", (0.0, 1.0)),
    "iou": ("max", (0.0, 1.0)),
    "miou": ("max", (0.0, 1.0)),
    "multiclass_jaccard_index": ("max", (0.0, 1.0)),
    "f1": ("max", (0.0, 1.0)),
    "multilabel_f1_score": ("max", (0.0, 1.0)),
    "auc": ("max", (0.0, 1.0)),
    "ap": ("max", (0.0, 1.0)),
    "map": ("max", (0.0, 1.0)),
    "test_test_map": ("max", (0.0, 1.0)),
    "test_test_segm_map": ("max", (0.0, 1.0)),
    "precision": ("max", (0.0, 1.0)),
    "recall": ("max", (0.0, 1.0)),
    "r2": ("max", (0.0, None)),
    "rmse": ("min", (0.0, None)),
    "mae": ("min", (0.0, None)),
    "mse": ("min", (0.0, None)),
    "mape": ("min", (0.0, None)),
    "logloss": ("min", (0.0, None)),
    "log_loss": ("min", (0.0, None)),
}

# Theoretical error optimum per metric, used to reconstruct raw error space
# independently of normalization bounds: bounded-[0,1] max metrics and r2 are
# perfect at 1.0; min metrics are perfect at 0.0.
_METRIC_ERROR_OPTIMUM = {
    "accuracy": 1.0,
    "acc": 1.0,
    "overall_accuracy": 1.0,
    "iou": 1.0,
    "miou": 1.0,
    "multiclass_jaccard_index": 1.0,
    "f1": 1.0,
    "multilabel_f1_score": 1.0,
    "auc": 1.0,
    "ap": 1.0,
    "map": 1.0,
    "test_test_map": 1.0,
    "test_test_segm_map": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "r2": 1.0,
    "rmse": 0.0,
    "mae": 0.0,
    "mse": 0.0,
    "mape": 0.0,
    "logloss": 0.0,
    "log_loss": 0.0,
}

_KNOWN_LIST = ", ".join(sorted(KNOWN_METRICS))


def get_direction(metric_name: str) -> str:
    """Return ``"min"`` or ``"max"`` for a known metric name.

    Args:
        metric_name: Metric identifier (case-insensitive).

    Returns:
        str: ``"min"`` (lower is better) or ``"max"`` (higher is better).

    Raises:
        ValueError: If ``metric_name`` is not in the registry. The message
            lists all known names and suggests using ``metric_direction`` for
            custom metrics.
    """
    name = metric_name.lower()
    if name not in KNOWN_METRICS:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Known metrics: {_KNOWN_LIST}. "
            f"For custom metrics, specify direction via metric_direction and "
            f"bounds via metric_type_bounds."
        )
    return KNOWN_METRICS[name][0]


def get_natural_bounds(metric_name: str) -> tuple:
    """Return ``(natural_low, natural_high)`` for a known metric name.

    ``natural_high`` is ``None`` for unbounded metrics (e.g. ``rmse``, ``mae``),
    meaning the upper bound must be supplied by the caller.

    Args:
        metric_name: Metric identifier (case-insensitive).

    Returns:
        tuple: ``(float, float | None)`` natural lower and upper bounds.

    Raises:
        ValueError: If ``metric_name`` is not in the registry.
    """
    name = metric_name.lower()
    if name not in KNOWN_METRICS:
        raise ValueError(
            f"Unknown metric '{metric_name}'. Known metrics: {_KNOWN_LIST}."
        )
    return KNOWN_METRICS[name][1]


def get_error_optimum(metric_name: str) -> float:
    """Return the theoretical error optimum for a known metric name.

    This is the metric's perfect-score value in raw score space (``1.0`` for
    bounded ``max`` metrics and ``r2``, ``0.0`` for ``min`` metrics), used to
    reconstruct raw error independently of normalization bounds.

    Args:
        metric_name: Metric identifier (case-insensitive).

    Returns:
        float: The theoretical optimum score for the metric.

    Raises:
        ValueError: If ``metric_name`` has no defined optimum. The message
            lists all known names and explains the current improvability
            fallback.
    """
    name = metric_name.lower()
    if name not in _METRIC_ERROR_OPTIMUM:
        raise ValueError(
            f"No error optimum defined for metric '{metric_name}'. Known "
            f"metrics: {_KNOWN_LIST}. For improvability, evaluma currently "
            "supports known registry metrics, or datasets explicitly marked "
            "metric_direction='min' so they are treated as error columns with "
            "optimum 0."
        )
    return _METRIC_ERROR_OPTIMUM[name]
