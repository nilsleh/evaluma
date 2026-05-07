import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from evaluma.results import IQMResult


def compute_iqm(raw_runs, norm_bounds, n_bootstrap=1000, random_state=None):
    """Compute Agarwal IQM on the flat run×dataset array with stratified bootstrap CIs.

    Implements the IQM from Agarwal et al. 2021 (rliable): trim the outer 25%
    of the concatenated per-dataset, per-seed normalized scores and average the
    remainder. Bootstrap CIs are stratified — seeds are resampled independently
    within each dataset stratum.

    Args:
        raw_runs: Long-format DataFrame with columns
            ``["model", "dataset", "seed", "score"]``.
        norm_bounds: ``(low, high, metric_direction)`` where ``low`` and
            ``high`` are per-dataset ``pd.Series`` of normalization bounds
            and ``metric_direction`` is a ``{dataset: "min"|"max"}`` dict
            (or ``None``).
        n_bootstrap: Number of stratified bootstrap replicates for the 95% CI.
        random_state: Seed for :func:`numpy.random.default_rng`.

    Returns:
        IQMResult: Result with ``.table`` sorted descending by IQM.
    """
    low, high, metric_direction = norm_bounds
    rng = np.random.default_rng(random_state)

    models = raw_runs["model"].unique().tolist()
    datasets = raw_runs["dataset"].unique().tolist()

    def _norm_score(score, dataset):
        lo = float(low[dataset])
        hi = float(high[dataset])
        if metric_direction and metric_direction.get(dataset) == "min":
            return (hi - score) / (hi - lo)
        return (score - lo) / (hi - lo)

    # Build per-(model, dataset) arrays of normalized scores
    per_dataset = {m: [] for m in models}
    for d in datasets:
        lo = float(low[d])
        hi = float(high[d])
        is_min = metric_direction and metric_direction.get(d) == "min"
        for m in models:
            mask = (raw_runs["model"] == m) & (raw_runs["dataset"] == d)
            scores = raw_runs.loc[mask, "score"].values.astype(float)
            if is_min:
                norm = (hi - scores) / (hi - lo)
            else:
                norm = (scores - lo) / (hi - lo)
            per_dataset[m].append(norm)

    # Point estimates: flat-array IQM
    iqms_arr = np.array(
        [trim_mean(np.concatenate(per_dataset[m]), 0.25) for m in models]
    )

    if n_bootstrap == 0:
        table = (
            pd.DataFrame(
                {"model": models, "IQM": iqms_arr, "CI_low": np.nan, "CI_high": np.nan}
            )
            .sort_values("IQM", ascending=False)
            .reset_index(drop=True)
        )
        return IQMResult(table)

    # Stratified bootstrap: resample seeds within each dataset independently
    boot_iqms = np.empty((n_bootstrap, len(models)))
    for b in range(n_bootstrap):
        for mi, m in enumerate(models):
            resampled = []
            for d_scores in per_dataset[m]:
                n = len(d_scores)
                idx = rng.integers(0, n, size=n)
                resampled.append(d_scores[idx])
            boot_iqms[b, mi] = trim_mean(np.concatenate(resampled), 0.25)

    ci_low = np.percentile(boot_iqms, 2.5, axis=0)
    ci_high = np.percentile(boot_iqms, 97.5, axis=0)

    table = (
        pd.DataFrame(
            {"model": models, "IQM": iqms_arr, "CI_low": ci_low, "CI_high": ci_high}
        )
        .sort_values("IQM", ascending=False)
        .reset_index(drop=True)
    )
    return IQMResult(table)
