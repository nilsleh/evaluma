"""Metric name registry: maps metric names to optimization direction and natural bounds."""

KNOWN_METRICS = {
    # name: (direction, (natural_low, natural_high))  — None high means user must specify
    "accuracy":  ("max", (0.0, 1.0)),
    "acc":       ("max", (0.0, 1.0)),
    "iou":       ("max", (0.0, 1.0)),
    "miou":      ("max", (0.0, 1.0)),
    "f1":        ("max", (0.0, 1.0)),
    "auc":       ("max", (0.0, 1.0)),
    "ap":        ("max", (0.0, 1.0)),
    "map":       ("max", (0.0, 1.0)),
    "precision": ("max", (0.0, 1.0)),
    "recall":    ("max", (0.0, 1.0)),
    "r2":        ("max", (0.0, 1.0)),
    "rmse":      ("min", (0.0, None)),
    "mae":       ("min", (0.0, None)),
    "mse":       ("min", (0.0, None)),
    "mape":      ("min", (0.0, None)),
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
