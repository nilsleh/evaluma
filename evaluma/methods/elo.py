from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_battles(
    scores: pd.DataFrame,
    tie_threshold: float | None = None,
) -> pd.DataFrame:
    """Generate pairwise battles from a normalized score matrix.

    For each dataset, all M*(M-1)/2 model pairs are compared. Every pair
    produces exactly one battle. Pairs within ``tie_threshold`` produce a tie
    (outcome=0.5). Each battle carries a weight of ``1 / n_battles_in_dataset``
    so that all datasets contribute equally to downstream fitting.

    Args:
        scores: Model × dataset normalized score matrix (higher = better).
        tie_threshold: Score difference at or below which a pair is called a
            tie (outcome=0.5). ``None`` means ties occur only on exact equality.

    Returns:
        DataFrame with columns ``["model_a", "model_b", "outcome", "dataset",
        "weight"]``. ``outcome=1.0`` when model_a wins, ``0.0`` when model_b
        wins, ``0.5`` for a tie.
    """
    models = scores.index.tolist()
    pairs = list(combinations(models, 2))
    n_pairs = len(pairs)

    rows = []
    for dataset in scores.columns:
        dataset_scores = scores[dataset]
        weight = 1.0 / n_pairs if n_pairs > 0 else 0.0
        for model_a, model_b in pairs:
            sa = float(dataset_scores[model_a])
            sb = float(dataset_scores[model_b])
            diff = sa - sb
            threshold = tie_threshold if tie_threshold is not None else 0.0
            if diff > threshold:
                rows.append((model_a, model_b, 1.0, dataset, weight))
            elif -diff > threshold:
                rows.append((model_a, model_b, 0.0, dataset, weight))
            else:
                rows.append((model_a, model_b, 0.5, dataset, weight))

    cols = ["model_a", "model_b", "outcome", "dataset", "weight"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def compute_winrate_matrix(
    scores: pd.DataFrame,
    tie_threshold: float | None = None,
) -> pd.DataFrame:
    """Compute M×M empirical win-rate matrix with equal dataset weighting.

    Args:
        scores: Model × dataset normalized score matrix.
        tie_threshold: Passed to :func:`compute_battles`.

    Returns:
        Square DataFrame (models × models). Cell (i,j) = fraction of datasets
        where model i beats model j. Diagonal is NaN. Rows sorted by
        descending average win-rate.
    """
    models = scores.index.tolist()
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    wins = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)

    for dataset in scores.columns:
        ds = scores[dataset]
        for model_a, model_b in combinations(models, 2):
            sa = float(ds[model_a])
            sb = float(ds[model_b])
            diff = sa - sb
            threshold = tie_threshold if tie_threshold is not None else 0.0
            i, j = idx[model_a], idx[model_b]
            if diff > threshold:
                wins[i, j] += 1.0
                counts[i, j] += 1.0
                counts[j, i] += 1.0
            elif -diff > threshold:
                wins[j, i] += 1.0
                counts[i, j] += 1.0
                counts[j, i] += 1.0
            else:
                wins[i, j] += 0.5
                wins[j, i] += 0.5
                counts[i, j] += 1.0
                counts[j, i] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        rates = np.where(counts > 0, wins / counts, np.nan)

    np.fill_diagonal(rates, np.nan)

    wm = pd.DataFrame(rates, index=models, columns=models)
    avg_wr = wm.mean(axis=1, skipna=True)
    order = avg_wr.sort_values(ascending=False).index
    return wm.loc[order, order]


def _fit_elo(
    battles: pd.DataFrame,
    models: list | None = None,
    scale: float = 400.0,
    base: float = 10.0,
    init_rating: float = 1000.0,
) -> pd.Series:
    """Fit MLE ELO ratings from a battles DataFrame.

    Uses logistic regression with per-battle sample weights. Falls back to
    iterative ELO (K=32) when logistic regression fails (e.g. total dominance).

    Args:
        battles: Output of :func:`compute_battles`.
        models: Explicit model list. When ``None``, inferred from battles.
            Must be provided when battles may be empty.
        scale: ELO scale parameter (default 400).
        base: Logarithm base (default 10).
        init_rating: Initial rating for iterative fallback (default 1000).

    Returns:
        pd.Series: ELO ratings indexed by model name.
    """
    from sklearn.linear_model import LogisticRegression

    if models is None:
        models = pd.concat([battles["model_a"], battles["model_b"]]).unique().tolist()

    if len(battles) == 0:
        return pd.Series(
            np.full(len(models), init_rating, dtype=np.float64), index=models
        )

    # sklearn LogisticRegression requires binary labels — expand tie battles
    # (outcome=0.5) into two half-weight battles with outcomes 1.0 and 0.0
    tie_mask = battles["outcome"] == 0.5
    if tie_mask.any():
        ties = battles[tie_mask].copy()
        half = ties.copy()
        ties["outcome"] = 1.0
        ties["weight"] = ties["weight"] / 2
        half["outcome"] = 0.0
        half["weight"] = half["weight"] / 2
        battles = pd.concat([battles[~tie_mask], ties, half], ignore_index=True)

    model_idx = {m: i for i, m in enumerate(models)}
    n = len(models)

    n_battles = len(battles)
    X = np.zeros((n_battles, n), dtype=np.float64)
    Y = battles["outcome"].values.astype(np.float64)
    W = battles["weight"].values.astype(np.float64)

    # Bake log(base) into the design matrix (TabArena convention) so the
    # logistic coefficients land directly on the ELO scale via `scale * coef`.
    log_base = np.log(base)
    for k, (_, row) in enumerate(battles.iterrows()):
        i = model_idx[row["model_a"]]
        j = model_idx[row["model_b"]]
        X[k, i] = log_base
        X[k, j] = -log_base

    try:
        unique_y = np.unique(Y)
        if len(unique_y) < 2:
            raise ValueError("single class")
        clf = LogisticRegression(fit_intercept=False, max_iter=1000, C=1e6)
        clf.fit(X, Y, sample_weight=W)
        # Convert logit coefficients to ELO scale (TabArena: elo = scale * coef).
        elo_ratings = clf.coef_[0] * scale
        elo_ratings = elo_ratings - elo_ratings.mean() + init_rating
    except Exception:
        logger.warning("MLE ELO failed; falling back to iterative ELO (K=32).")
        elo_ratings = _iterative_elo(battles, model_idx, n, scale, base, init_rating)

    return pd.Series(elo_ratings, index=models)


def _iterative_elo(
    battles: pd.DataFrame,
    model_idx: dict,
    n: int,
    scale: float,
    base: float,
    init_rating: float,
    K: float = 32.0,
) -> np.ndarray:
    """Iterative ELO update rule fallback."""
    ratings = np.full(n, init_rating, dtype=np.float64)
    for _, row in battles.iterrows():
        i = model_idx[row["model_a"]]
        j = model_idx[row["model_b"]]
        ea = 1.0 / (1.0 + base ** ((ratings[j] - ratings[i]) / scale))
        eb = 1.0 - ea
        outcome = float(row["outcome"])
        ratings[i] += K * (outcome - ea)
        ratings[j] += K * ((1.0 - outcome) - eb)
    return ratings


def compute_battles_from_runs(
    raw_runs: pd.DataFrame,
    tie_threshold: float | None = None,
    metric_direction: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Generate battles from per-seed long-format data.

    Pairs all models within each (dataset, seed) combination. Each dataset
    contributes total weight 1 regardless of seed count. Datasets mapped to
    ``"min"`` in ``metric_direction`` are negated so that higher score always
    means better throughout.

    Args:
        raw_runs: Long-format DataFrame with columns
            ``["model", "dataset", "seed", "score"]``.
        tie_threshold: Score difference at or below which a pair is a tie.
        metric_direction: Maps dataset names to ``"min"`` or ``"max"``.
            Datasets mapped to ``"min"`` are negated before comparison.

    Returns:
        DataFrame with columns ``["model_a", "model_b", "outcome", "dataset",
        "seed", "weight"]``.
    """
    rows = []
    models = raw_runs["model"].unique().tolist()
    pairs = list(combinations(models, 2))
    n_pairs = len(pairs)

    for dataset, ds_df in raw_runs.groupby("dataset"):
        sign = -1.0 if (metric_direction or {}).get(dataset) == "min" else 1.0
        seeds = ds_df["seed"].unique()
        n_seeds = len(seeds)
        weight = 1.0 / (n_pairs * n_seeds) if n_pairs > 0 else 0.0
        for seed, seed_df in ds_df.groupby("seed"):
            seed_scores = seed_df.set_index("model")["score"] * sign
            for model_a, model_b in pairs:
                if model_a not in seed_scores.index or model_b not in seed_scores.index:
                    continue
                sa = float(seed_scores[model_a])
                sb = float(seed_scores[model_b])
                diff = sa - sb
                threshold = tie_threshold if tie_threshold is not None else 0.0
                if diff > threshold:
                    outcome = 1.0
                elif -diff > threshold:
                    outcome = 0.0
                else:
                    outcome = 0.5
                rows.append((model_a, model_b, outcome, dataset, seed, weight))

    cols = ["model_a", "model_b", "outcome", "dataset", "seed", "weight"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def compute_elo(
    scores: pd.DataFrame,
    n_bootstrap: int = 1000,
    random_state=None,
    tie_threshold: float | None = None,
    calibration_model: str | None = None,
    raw_runs: pd.DataFrame | None = None,
    metric_direction: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute MLE ELO ratings with battle-within-task bootstrap CIs.

    Args:
        scores: Model × dataset normalized score matrix (higher = better).
        n_bootstrap: Number of bootstrap replicates for 95% CI. Set to 0 to
            skip bootstrap (CI columns will be NaN).
        random_state: Seed for :func:`numpy.random.default_rng`.
        tie_threshold: Minimum score difference to emit a non-tie battle.
        calibration_model: If given, shift all ratings so this model has
            ELO = 1000.
        raw_runs: Long-format per-seed DataFrame with columns
            ``["model", "dataset", "seed", "score"]``. When provided, battles
            are generated per (dataset, seed) via
            :func:`compute_battles_from_runs` and bootstrap resamples within
            (dataset, seed) groups.
        metric_direction: Passed to :func:`compute_battles_from_runs`. Ignored
            when ``raw_runs`` is ``None``.

    Returns:
        Tuple of:
        - DataFrame with columns ``["model", "ELO", "CI_low", "CI_high"]``
          sorted descending by ELO.
        - M×M win-rate matrix as returned by :func:`compute_winrate_matrix`.

    Raises:
        ValueError: If ``calibration_model`` is not in ``scores.index``.
    """
    if calibration_model is not None and calibration_model not in scores.index:
        raise ValueError(
            f"calibration_model={calibration_model!r} not found in score matrix. "
            f"Available models: {scores.index.tolist()}"
        )

    all_models = scores.index.tolist()

    if raw_runs is not None:
        battles = compute_battles_from_runs(
            raw_runs, tie_threshold=tie_threshold, metric_direction=metric_direction
        )
    else:
        battles = compute_battles(scores, tie_threshold=tie_threshold)

    winrate_matrix = compute_winrate_matrix(scores, tie_threshold=tie_threshold)

    point_ratings = _fit_elo(battles, models=all_models)

    if calibration_model is not None:
        shift = 1000.0 - float(point_ratings[calibration_model])
        point_ratings = point_ratings + shift

    if n_bootstrap == 0:
        models = point_ratings.index.tolist()
        table = (
            pd.DataFrame(
                {
                    "model": models,
                    "ELO": point_ratings.values,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                }
            )
            .sort_values("ELO", ascending=False)
            .reset_index(drop=True)
        )
        return table, winrate_matrix

    rng = np.random.default_rng(random_state)
    models = point_ratings.index.tolist()

    # Resample within (dataset, seed) groups when raw_runs is used; else by dataset
    group_cols = ["dataset", "seed"] if "seed" in battles.columns else ["dataset"]

    boot_ratings = np.empty((n_bootstrap, len(models)))
    for b in range(n_bootstrap):
        resampled_parts = []
        for _, subset in battles.groupby(group_cols):
            if len(subset) == 0:
                continue
            idx = rng.integers(0, len(subset), size=len(subset))
            resampled_parts.append(subset.iloc[idx])
        if not resampled_parts:
            boot_ratings[b] = point_ratings.values
            continue
        boot_battles = pd.concat(resampled_parts, ignore_index=True)
        boot_r = _fit_elo(boot_battles, models=models)
        boot_r_aligned = boot_r.reindex(models).fillna(1000.0)
        if calibration_model is not None:
            shift = 1000.0 - float(boot_r_aligned[calibration_model])
            boot_r_aligned = boot_r_aligned + shift
        boot_ratings[b] = boot_r_aligned.values

    ci_low = np.percentile(boot_ratings, 2.5, axis=0)
    ci_high = np.percentile(boot_ratings, 97.5, axis=0)

    table = (
        pd.DataFrame(
            {
                "model": models,
                "ELO": point_ratings.values,
                "CI_low": ci_low,
                "CI_high": ci_high,
            }
        )
        .sort_values("ELO", ascending=False)
        .reset_index(drop=True)
    )
    return table, winrate_matrix
