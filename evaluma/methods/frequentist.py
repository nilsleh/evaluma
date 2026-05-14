import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import chi2 as chi2_dist
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

from evaluma.results import FrequentistResult


def _holm_correction(p_values: list[float]) -> np.ndarray:
    """Holm (1979) step-down correction. Verified against statsmodels in tests."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = sorted_p * np.arange(n, 0, -1)
    adjusted = np.minimum(np.maximum.accumulate(adjusted), 1.0)
    result = np.empty(n)
    result[order] = adjusted
    return result


def compute_frequentist(
    scores_matrix: pd.DataFrame,
    *,
    reference=None,
    alpha=0.05,
) -> FrequentistResult:
    """Compute frequentist model comparisons.

    All-pairs mode follows the Demšar (2006) / autorank Friedman + Nemenyi
    workflow. Reference mode is an evaluma extension: pairwise Wilcoxon
    signed-rank tests against a named baseline with Holm correction.

    Always runs a Friedman omnibus test first. In all-pairs mode, follows with
    Nemenyi post-hoc and computes the critical difference (CD) scalar. In
    reference mode, follows with Wilcoxon signed-rank + Holm correction against
    one reference model.

    Note: For k=2, Friedman+Nemenyi is used uniformly rather than the standalone
    Wilcoxon special-case from Demšar (2006). This is slightly less powerful at
    small N (5–10) but avoids branching complexity and is statistically valid.

    Args:
        scores_matrix: Normalized model × dataset score matrix (models as row
            index, datasets as columns).
        reference: If provided, only compare every other model against this one.
            ``None`` triggers all-pairs mode.
        alpha: Significance level (default 0.05).

    Returns:
        FrequentistResult

    Raises:
        ValueError: If k < 2, N < 5, or reference not found in scores_matrix.

    References:
        Demšar, J. (2006). Statistical Comparisons of Classifiers over
        Multiple Data Sets. *JMLR*, 7, 1–30.
    """
    k = len(scores_matrix.index)
    N = scores_matrix.shape[1]
    if k < 2:
        raise ValueError("Need at least 2 models to compare.")
    if N < 5:
        raise ValueError(f"Need at least 5 datasets; got {N}.")
    if reference is not None and reference not in scores_matrix.index:
        raise ValueError(f"Reference model {reference!r} not found in scores matrix.")

    models = scores_matrix.index.tolist()

    # rank 1 = best (highest normalized score) per dataset
    ranked = scores_matrix.rank(ascending=False, axis=0)
    avg_ranks = ranked.mean(axis=1).sort_values()

    # Friedman omnibus test — always run first in both modes.
    # scipy.friedmanchisquare requires k≥3; for k=2 compute chi-squared manually.
    if k >= 3:
        friedman_stat, friedman_p = friedmanchisquare(
            *[scores_matrix.loc[m].values for m in models]
        )
        friedman_stat = float(friedman_stat)
        friedman_p = float(friedman_p)
    else:
        # k=2: rank ascending within each dataset, compute standard Friedman formula
        # ascending=True: Friedman formula convention (rank 1 = lowest score)
        ranked_f = scores_matrix.rank(ascending=True, axis=0)
        R = ranked_f.sum(axis=1).values
        friedman_stat = float(
            12.0 / (N * k * (k + 1)) * float((R ** 2).sum()) - 3 * N * (k + 1)
        )
        friedman_p = float(chi2_dist.sf(friedman_stat, df=k - 1))
    if friedman_p >= alpha:
        warnings.warn(
            f"Friedman test not significant (p={friedman_p:.3g}); "
            "post-hoc results may not be valid.",
            UserWarning,
            stacklevel=2,
        )

    if reference is None:
        # All-pairs: Nemenyi post-hoc
        # posthoc_nemenyi_friedman expects datasets × models
        ph = sp.posthoc_nemenyi_friedman(scores_matrix.T)
        rows = []
        for a, b in combinations(models, 2):
            rows.append({
                "model_a": a,
                "model_b": b,
                "rank_diff": abs(avg_ranks[a] - avg_ranks[b]),
                "p_value": float(ph.loc[a, b]),
                "significant": bool(ph.loc[a, b] < alpha),
            })

        # CD scalar — use df=np.inf to match posthoc_nemenyi_friedman exactly
        q_alpha = studentized_range.ppf(1 - alpha, k, df=np.inf) / np.sqrt(2)
        cd = float(q_alpha * np.sqrt(k * (k + 1) / (6 * N)))

        return FrequentistResult(
            pd.DataFrame(rows),
            avg_ranks=avg_ranks,
            friedman_statistic=float(friedman_stat),
            friedman_p_value=float(friedman_p),
            reference=None,
            alpha=alpha,
            cd=cd,
        )
    else:
        # Reference mode: Wilcoxon + Holm
        pair_list = [(reference, m) for m in models if m != reference]
        rows = []
        for ref, other in pair_list:
            diff = scores_matrix.loc[ref].values - scores_matrix.loc[other].values
            if np.all(diff == 0):
                W, p = 0.0, 1.0
            else:
                res = wilcoxon(diff, alternative="two-sided")
                W, p = float(res.statistic), float(res.pvalue)
            rows.append({"model_a": ref, "model_b": other, "w_statistic": W, "p_value": p})

        raw_p = [r["p_value"] for r in rows]
        corrected = _holm_correction(raw_p)
        for i, row in enumerate(rows):
            row["p_value_corrected"] = float(corrected[i])
            row["significant"] = bool(corrected[i] < alpha)

        return FrequentistResult(
            pd.DataFrame(rows),
            avg_ranks=avg_ranks,
            friedman_statistic=float(friedman_stat),
            friedman_p_value=float(friedman_p),
            reference=reference,
            alpha=alpha,
            cd=None,
        )
