import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from evaluma.results import AggregateResult

_AGG_MODES = {"trimmed_mean", "mean", "median"}


def _aggregate_scores(scores_matrix: pd.DataFrame, agg="trimmed_mean") -> pd.Series:
    """Collapse a model × dataset matrix to one score per model.

    Shared primitive for the aggregate-ranking and rank-sensitivity paths so the
    two cannot drift in how they define a model's summary score.

    Args:
        scores_matrix: Normalized model × dataset score matrix.
        agg: Aggregation mode — one of ``"trimmed_mean"``, ``"mean"``,
            ``"median"``.

    Returns:
        pd.Series: Per-model scores indexed by ``scores_matrix.index``.

    Raises:
        ValueError: If ``agg`` is not one of the supported modes.
    """
    if agg not in _AGG_MODES:
        raise ValueError(f"agg must be one of {sorted(_AGG_MODES)!r}, got {agg!r}")

    data = scores_matrix.values  # shape (n_models, n_datasets)
    if agg == "trimmed_mean":
        scores = np.array([trim_mean(row, proportiontocut=0.25) for row in data])
    elif agg == "mean":
        scores = data.mean(axis=1)
    else:
        scores = np.median(data, axis=1)

    return pd.Series(scores, index=scores_matrix.index)


def compute_aggregate(
    scores_matrix: pd.DataFrame, agg="trimmed_mean"
) -> AggregateResult:
    """Compute a point-estimate descriptive ranking from a normalized score matrix.

    .. note::
        This is a **descriptive point estimate only** (no CI). The
        trimmed-mean variant trims across datasets, not across seeds; with
        fewer than ~10 datasets the 25% trim is aggressive (e.g. 5 datasets
        → only 3 contribute). Treat results as exploratory. For a
        statistically grounded ranking with uncertainty, use
        :func:`evaluma.methods.iqm.compute_iqm` (requires multiple seeds).

    Args:
        scores_matrix: Normalized model × dataset score matrix.
        agg: Aggregation mode — one of ``"trimmed_mean"``, ``"mean"``,
            ``"median"``.

    Returns:
        AggregateResult: Result with ``.table`` sorted descending by ``score``.

    Raises:
        ValueError: If ``agg`` is not one of the supported modes.
    """
    scores = _aggregate_scores(scores_matrix, agg=agg)
    table = (
        pd.DataFrame({"model": scores.index.tolist(), "score": scores.to_numpy()})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    return AggregateResult(table)
