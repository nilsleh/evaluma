import numpy as np
import pandas as pd

from evaluma.results import ProfileResult


def compute_profiles(
    scores_matrix: pd.DataFrame,
    metric_direction: dict | None = None,
) -> ProfileResult:
    """Compute Dolan-Moré performance profiles.

    Ratios are computed directly from raw scores without normalization:

    - **max metrics** (higher is better): r_ij = best_j / score_ij
    - **min metrics** (lower is better): r_ij = score_ij / best_j

    where best_j is the best score across all models on dataset j. Both
    definitions yield r_ij ≥ 1, with r_ij = 1 when model i is the best on
    dataset j.

    The tau grid uses the exact observed ratio values as breakpoints, giving
    the exact Dolan-Moré step function. The plot renders on a log₁₀(τ) axis,
    following ML-GYM (Batra et al., 2025) and the AutoML Decathlon (Roberts
    et al., 2022), which extended Dolan & Moré (2002). Use
    ``ProfileResult.aup`` for the scalar Area Under the Profile summary.

    Args:
        scores_matrix: Raw model × dataset score matrix. All values must be
            strictly positive.
        metric_direction: Dict mapping dataset (column) names to ``"min"`` or
            ``"max"``. Datasets absent from the dict default to ``"max"``.

    Returns:
        ProfileResult with ``.table`` (long-format: ``tau``, ``model``,
        ``fraction_within_tau``).

    Raises:
        ValueError: If any value in ``scores_matrix`` is zero or negative.
    """
    if (scores_matrix.values <= 0).any():
        raise ValueError(
            "Cannot compute performance profiles: score of 0 or below detected. "
            "All scores must be strictly positive."
        )

    metric_direction = metric_direction or {}
    models = scores_matrix.index.tolist()
    data = scores_matrix.values.astype(float)  # (n_models, n_datasets)
    datasets = scores_matrix.columns.tolist()

    # Compute per-dataset ratios respecting direction.
    ratios = np.ones_like(data)
    for j, dataset in enumerate(datasets):
        col = data[:, j]
        if metric_direction.get(dataset) == "min":
            best = col.min()
            ratios[:, j] = col / best
        else:
            best = col.max()
            ratios[:, j] = best / col

    # Exact breakpoints: every observed ratio value, plus 1.0 as anchor.
    tau_grid = np.sort(np.unique(np.concatenate([[1.0], ratios.flatten()])))

    rows = []
    for tau in tau_grid:
        for i, model in enumerate(models):
            frac = (ratios[i] <= tau).mean()
            rows.append({"tau": tau, "model": model, "fraction_within_tau": frac})

    return ProfileResult(pd.DataFrame(rows))
