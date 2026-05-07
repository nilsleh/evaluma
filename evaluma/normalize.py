import warnings

import pandas as pd


def normalize(matrix, *, norm_ref_low=None, norm_ref_high=None, metric_direction=None):
    """Apply per-dataset min-max normalization to a score matrix.

    Args:
        matrix: Model × dataset score matrix (models as rows, datasets as
            columns).
        norm_ref_low: Lower bound for normalization — scalar, model name
            (row label), or ``{dataset: value}`` dict. ``None`` uses the
            per-dataset observed minimum and emits a ``UserWarning``.
        norm_ref_high: Upper bound for normalization, same format as
            ``norm_ref_low``. ``None`` uses the per-dataset observed
            maximum.
        metric_direction: Dict mapping dataset names to ``"min"`` or
            ``"max"``. Entries mapped to ``"min"`` cause the matrix to be
            negated before normalization.

    Returns:
        pandas.DataFrame: Normalized matrix with the same shape and index
        as ``matrix``, values in ``[0, 1]`` within the reference bounds.

    Raises:
        ValueError: If ``norm_ref_low`` or ``norm_ref_high`` is a string
            that does not name a row in ``matrix``.
    """
    mat = matrix.copy().astype(float)

    emit_warning = norm_ref_low is None and norm_ref_high is None

    # Resolve bounds on the original matrix before any direction inversion.
    low = _resolve_bound(mat, norm_ref_low, use_min=True)
    high = _resolve_bound(mat, norm_ref_high, use_min=False)

    # Per-column direction inversion: negate "min" columns and flip their bounds
    # so that a lower original score maps to a higher normalized score.
    if metric_direction is not None:
        for col, direction in metric_direction.items():
            if direction == "min" and col in mat.columns:
                mat[col] = -mat[col]
                low[col], high[col] = -high[col], -low[col]

    if emit_warning:
        warnings.warn(
            "Normalization bounds depend on observed data; "
            "pass norm_ref_low/norm_ref_high for stable bounds.",
            UserWarning,
            stacklevel=2,
        )

    return (mat - low) / (high - low)


def _resolve_bound(mat, bound, use_min):
    """Resolve a normalization bound specification to a per-column Series.

    Args:
        mat: Score matrix (models × datasets).
        bound: ``None`` (use data min/max), a scalar, a model-name string,
            or a ``{dataset: value}`` dict.
        use_min: When ``bound`` is ``None``, return the column-wise minimum
            if ``True``, maximum if ``False``.

    Returns:
        pandas.Series: Per-column (dataset) bound values.

    Raises:
        ValueError: If ``bound`` is a string not present in ``mat.index``.
    """
    if bound is None:
        return mat.min() if use_min else mat.max()
    if isinstance(bound, pd.Series):
        return bound
    if isinstance(bound, str):
        if bound not in mat.index:
            raise ValueError(f"Reference model '{bound}' not found in score matrix")
        return mat.loc[bound]
    if isinstance(bound, dict):
        return pd.Series({col: bound[col] for col in mat.columns})
    # Scalar
    return pd.Series({col: float(bound) for col in mat.columns})
