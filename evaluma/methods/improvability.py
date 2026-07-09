import numpy as np
import pandas as pd

from evaluma.results import ImprovabilityResult


def _compute_error_matrix(
    raw_matrix: pd.DataFrame, direction_map: dict, optimum_map: dict
) -> pd.DataFrame:
    """Reconstruct raw error from a dense score matrix and metric metadata.

    Per column, error is ``optimum - score`` for ``max`` metrics and
    ``score - optimum`` for ``min`` metrics, so higher error is always worse.
    Assumes a dense (complete) matrix.

    Args:
        raw_matrix: Model × dataset raw score matrix.
        direction_map: Dataset → ``"min"`` / ``"max"`` metric direction.
        optimum_map: Dataset → theoretical error optimum (perfect score).

    Returns:
        pd.DataFrame: Error matrix with the same shape, index, and columns.
    """
    error = raw_matrix.astype(float).copy()
    for col in raw_matrix.columns:
        optimum = optimum_map[col]
        if direction_map[col] == "max":
            error[col] = optimum - raw_matrix[col]
        else:
            error[col] = raw_matrix[col] - optimum
    return error


def compute_improvability(
    raw_matrix: pd.DataFrame, direction_map: dict, optimum_map: dict
) -> ImprovabilityResult:
    """Compute mean improvability per model in raw error space.

    Reconstructs raw error, takes the per-dataset best error over models, and
    computes per-cell ``(error - best_err) / error * 100`` (percent error
    reduction needed to match the best method). A zero-error cell is already
    optimal, so its improvability is ``0.0`` (avoiding division by zero). Mean
    improvability averages over datasets. Faithful to the TabArena /
    BeyondArena definition.

    Args:
        raw_matrix: Model × dataset raw score matrix (dense).
        direction_map: Dataset → ``"min"`` / ``"max"`` metric direction.
        optimum_map: Dataset → theoretical error optimum.

    Returns:
        ImprovabilityResult: ``.table`` (``model``, ``improvability``; sorted
            ascending) and ``.per_dataset`` diagnostic long table.
    """
    error = _compute_error_matrix(raw_matrix, direction_map, optimum_map)
    best_err = error.min(axis=0)  # per dataset, over models

    with np.errstate(divide="ignore", invalid="ignore"):
        imp = (error - best_err) / error * 100.0
    imp = imp.where(error != 0, 0.0)  # error == 0 -> already optimal -> 0.0

    mean_imp = imp.mean(axis=1)
    table = (
        pd.DataFrame(
            {
                "model": mean_imp.index.tolist(),
                "improvability": mean_imp.to_numpy(),
            }
        )
        .sort_values("improvability")
        .reset_index(drop=True)
    )

    err_long = error.stack()
    per_dataset = pd.DataFrame(
        {
            "model": err_long.index.get_level_values(0),
            "dataset": err_long.index.get_level_values(1),
            "error": err_long.to_numpy(),
            "improvability": imp.stack().reindex(err_long.index).to_numpy(),
        }
    )
    per_dataset["best_error"] = per_dataset["dataset"].map(best_err)
    per_dataset = per_dataset[
        ["model", "dataset", "error", "best_error", "improvability"]
    ]

    return ImprovabilityResult(table, per_dataset)
