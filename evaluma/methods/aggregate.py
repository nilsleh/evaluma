import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from evaluma.results import AggregateResult

_AGG_MODES = {"trimmed_mean", "mean", "median"}


def compute_aggregate(
    scores_matrix: pd.DataFrame, agg="trimmed_mean"
) -> AggregateResult:
    """Compute a point-estimate descriptive ranking from a normalized score matrix.

    Args:
        scores_matrix: Normalized model × dataset score matrix.
        agg: Aggregation mode — one of ``"trimmed_mean"``, ``"mean"``,
            ``"median"``.

    Returns:
        AggregateResult: Result with ``.table`` sorted descending by ``score``.

    Raises:
        ValueError: If ``agg`` is not one of the supported modes.
    """
    if agg not in _AGG_MODES:
        raise ValueError(f"agg must be one of {sorted(_AGG_MODES)!r}, got {agg!r}")

    models = scores_matrix.index.tolist()
    data = scores_matrix.values  # shape (n_models, n_datasets)

    if agg == "trimmed_mean":
        scores = np.array([trim_mean(row, proportiontocut=0.25) for row in data])
    elif agg == "mean":
        scores = data.mean(axis=1)
    else:
        scores = np.median(data, axis=1)

    table = (
        pd.DataFrame({"model": models, "score": scores})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return AggregateResult(table)
