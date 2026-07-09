import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from evaluma.methods.aggregate import _aggregate_scores
from evaluma.results import RankSensitivityResult


def _validate_aligned_axes(scores_a: pd.DataFrame, scores_b: pd.DataFrame):
    """Validate that model and dataset label sets match across conditions.

    Args:
        scores_a: Condition A normalized scores (model × dataset).
        scores_b: Condition B normalized scores (model × dataset).

    Raises:
        ValueError: If model label sets differ.
        ValueError: If dataset label sets differ.
    """
    models_a = set(scores_a.index)
    models_b = set(scores_b.index)
    datasets_a = set(scores_a.columns)
    datasets_b = set(scores_b.columns)

    missing_in_b = sorted(models_a - models_b)
    missing_in_a = sorted(models_b - models_a)
    if missing_in_a or missing_in_b:
        parts = []
        if missing_in_b:
            parts.append(f"missing from condition B: {missing_in_b}")
        if missing_in_a:
            parts.append(f"missing from condition A: {missing_in_a}")
        raise ValueError("Model mismatch: " + "; ".join(parts))

    missing_ds_in_b = sorted(datasets_a - datasets_b)
    missing_ds_in_a = sorted(datasets_b - datasets_a)
    if missing_ds_in_a or missing_ds_in_b:
        parts = []
        if missing_ds_in_b:
            parts.append(f"missing from condition B: {missing_ds_in_b}")
        if missing_ds_in_a:
            parts.append(f"missing from condition A: {missing_ds_in_a}")
        raise ValueError("Dataset mismatch: " + "; ".join(parts))


def _ranks_from_scores(scores: pd.DataFrame, agg="trimmed_mean") -> pd.Series:
    """Aggregate per-model scores and convert them to ranks.

    Args:
        scores: Normalized score matrix (model × dataset).
        agg: Aggregation mode passed to
            :func:`~evaluma.methods.aggregate._aggregate_scores`; defaults to
            ``"trimmed_mean"`` to match ``aggregate_ranking``.

    Returns:
        pd.Series: Average ranks with rank 1 as best (higher score is better).
    """
    return _aggregate_scores(scores, agg=agg).rank(ascending=False, method="average")


def _delta_table(
    rank_a: pd.Series, rank_b: pd.Series, cond_a_str: str, cond_b_str: str
) -> pd.DataFrame:
    """Build the sorted per-model ``delta_rank`` table from two rank vectors.

    Args:
        rank_a: Condition A per-model ranks (rank 1 = best), indexed by model.
        rank_b: Condition B per-model ranks, aligned to ``rank_a``'s index.
        cond_a_str: Condition A label (used in the ``rank_{cond_a}`` column).
        cond_b_str: Condition B label.

    Returns:
        pd.DataFrame with columns ``model``, ``rank_{cond_a}``,
        ``rank_{cond_b}``, ``delta_rank``, sorted by descending
        ``|delta_rank|`` then model.
    """
    rank_a_col = f"rank_{cond_a_str}"
    rank_b_col = f"rank_{cond_b_str}"
    table = pd.DataFrame(
        {
            "model": rank_a.index.tolist(),
            rank_a_col: rank_a.values.astype(float),
            rank_b_col: rank_b.values.astype(float),
        }
    )
    table["delta_rank"] = table[rank_b_col] - table[rank_a_col]
    return (
        table.assign(_abs_delta=table["delta_rank"].abs())
        .sort_values(["_abs_delta", "model"], ascending=[False, True])
        .drop(columns="_abs_delta")
        .reset_index(drop=True)
    )


def compute_rank_sensitivity_from_ranks(
    rank_a: pd.Series,
    rank_b: pd.Series,
    cond_a,
    cond_b,
    agg: str = "ranker",
) -> RankSensitivityResult:
    """Point-estimate rank sensitivity from two precomputed rank vectors.

    Decouples the ranking step from the tau computation so any ranking method
    (average rank, ELO, improvability, or a custom callable) can drive the
    sensitivity readout — not only the ``_aggregate_scores`` family. The
    inputs are assumed to be literal rank vectors, not arbitrary score-like
    keys. No bootstrap CI is produced (``tau_ci`` is ``(nan, nan)``).

    Args:
        rank_a: Condition A per-model ranks (rank 1 = best), indexed by model.
        rank_b: Condition B per-model ranks; must cover ``rank_a``'s roster.
        cond_a: Label for condition A.
        cond_b: Label for condition B.
        agg: Label describing the ranking provenance (stored on the result for
            provenance / plot titles).

    Returns:
        RankSensitivityResult: Point estimates (``tau``, ``rho``), the
        ``delta_rank`` table, and ``tau_ci=(nan, nan)``.
    """
    rank_b = rank_b.loc[rank_a.index]
    tau = float(kendalltau(rank_a.values, rank_b.values, method="auto").statistic)
    rho = float(spearmanr(rank_a.values, rank_b.values).statistic)
    table = _delta_table(rank_a, rank_b, str(cond_a), str(cond_b))
    return RankSensitivityResult(
        tau=tau,
        tau_ci=(np.nan, np.nan),
        rho=rho,
        table=table,
        cond_a=str(cond_a),
        cond_b=str(cond_b),
        agg=agg,
    )


def compute_rank_sensitivity(
    scores_a: pd.DataFrame,
    scores_b: pd.DataFrame,
    cond_a,
    cond_b,
    n_bootstrap=1000,
    random_state=None,
    agg="trimmed_mean",
) -> RankSensitivityResult:
    """Compute ranking sensitivity between two model×dataset score matrices.

    Args:
        scores_a: Condition A normalized scores (model × dataset).
        scores_b: Condition B normalized scores (model × dataset).
        cond_a: Label for condition A (used in output table/plot labels).
        cond_b: Label for condition B (used in output table/plot labels).
        n_bootstrap: Number of dataset-bootstrap samples for the 95% CI.
        random_state: Seed for ``numpy.random.default_rng``.
        agg: Per-model aggregation defining the ranking — ``"trimmed_mean"``
            (default, matching ``aggregate_ranking``), ``"mean"``, or
            ``"median"``.

    Returns:
        RankSensitivityResult: Rank sensitivity point estimates, CI, and table.

    Raises:
        ValueError: If ``n_bootstrap < 0``.
        ValueError: If ``agg`` is not a supported mode.
        ValueError: If model or dataset labels are misaligned.
    """
    if n_bootstrap < 0:
        raise ValueError(f"n_bootstrap must be >= 0, got {n_bootstrap}.")

    _validate_aligned_axes(scores_a, scores_b)

    scores_b = scores_b.loc[scores_a.index, scores_a.columns]
    rank_a = _ranks_from_scores(scores_a, agg=agg)
    rank_b = _ranks_from_scores(scores_b, agg=agg)

    tau = float(kendalltau(rank_a.values, rank_b.values, method="auto").statistic)
    rho = float(spearmanr(rank_a.values, rank_b.values).statistic)

    cond_a_str = str(cond_a)
    cond_b_str = str(cond_b)
    table = _delta_table(rank_a, rank_b, cond_a_str, cond_b_str)

    if n_bootstrap == 0:
        tau_ci = (np.nan, np.nan)
        return RankSensitivityResult(
            tau=tau,
            tau_ci=tau_ci,
            rho=rho,
            table=table,
            cond_a=cond_a_str,
            cond_b=cond_b_str,
            agg=agg,
        )

    rng = np.random.default_rng(random_state)
    n_datasets = scores_a.shape[1]
    boot_tau = np.full(n_bootstrap, np.nan, dtype=float)
    columns = scores_a.columns.to_numpy()

    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n_datasets, size=n_datasets)
        sampled_cols = columns[sample_idx]
        rank_a_b = _ranks_from_scores(scores_a.loc[:, sampled_cols], agg=agg)
        rank_b_b = _ranks_from_scores(scores_b.loc[:, sampled_cols], agg=agg)
        boot_tau[i] = kendalltau(
            rank_a_b.values, rank_b_b.values, method="auto"
        ).statistic

    finite = np.isfinite(boot_tau)
    if not finite.any():
        warnings.warn(
            "All bootstrap rank-correlation replicates were undefined; "
            "returning tau_ci=(nan, nan).",
            UserWarning,
            stacklevel=2,
        )
        tau_ci = (np.nan, np.nan)
    else:
        ci = np.nanpercentile(boot_tau, [2.5, 97.5])
        tau_ci = (float(ci[0]), float(ci[1]))

    return RankSensitivityResult(
        tau=tau,
        tau_ci=tau_ci,
        rho=rho,
        table=table,
        cond_a=cond_a_str,
        cond_b=cond_b_str,
        agg=agg,
    )
