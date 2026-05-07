import pandas as pd

from evaluma._version import __version__  # noqa: F401
from evaluma.benchmark import Benchmark  # noqa: F401


def load_df(
    df,
    *,
    model="model",
    dataset="dataset",
    metric="metric",
    score="score",
    seed=None,
    metric_type_bounds=None,
    norm_ref_low=None,
    norm_ref_high=None,
    metric_direction=None,
    drop_incomplete=False,
):
    """Load a DataFrame and return a ready-to-use Benchmark object.

    Args:
        df: A pandas DataFrame in long format (one row per model/dataset pair).
            To load from a CSV file, use :func:`evaluma.load_csv` instead.
        model: Column name for the model identifier.
        dataset: Column name for the dataset identifier.
        metric: Column name for the metric identifier.
        score: Column name for the score values.
        seed: Column name for the random seed. When provided, all seed rows
            are preserved and a ``seed`` column is included in the loaded
            DataFrame.
        metric_type_bounds: Dict mapping metric names to ``(low, high)``
            bound tuples. ``high`` may be a model name string, resolved
            per-dataset to that model's score. When provided, ``norm_ref_low``
            and ``norm_ref_high`` must be ``None``. Metric direction is inferred
            from the built-in registry; unknown metrics raise ``ValueError``.
            Bounded metrics not listed here (e.g. accuracy, iou, f1) use their
            natural ``[0, 1]`` bounds automatically. Unbounded metrics (rmse,
            mae, mse) must be listed; omitting them raises ``ValueError``.
        norm_ref_low: Lower normalization reference — scalar, model name,
            or per-dataset dict. If ``None``, the per-dataset minimum is
            used and a ``UserWarning`` is emitted. Cannot be combined with
            ``metric_type_bounds``.
        norm_ref_high: Upper normalization reference, same format as
            ``norm_ref_low``. If ``None``, the per-dataset maximum is used.
            Cannot be combined with ``metric_type_bounds``.
        metric_direction: Dict mapping dataset names to ``"min"`` or
            ``"max"``. When used with ``metric_type_bounds``, these entries
            take precedence over the registry-inferred direction. Without
            ``metric_type_bounds``, datasets mapped to ``"min"`` are negated
            before normalization so that higher is always better.
        drop_incomplete: If ``True``, silently drop models with missing
            scores instead of raising.

    Returns:
        Benchmark: Normalized benchmark ready for analysis.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If ``metric_type_bounds`` is provided together with
            ``norm_ref_low`` or ``norm_ref_high``.
        ValueError: If the data contains more than one metric per
            (model, dataset) pair, or if the score matrix is incomplete
            and ``drop_incomplete`` is ``False``.
        ValueError: If a metric referenced by a dataset is not in the
            registry and not covered by ``metric_type_bounds``.
        ValueError: If a regression metric (rmse, mae, mse) is present but
            no upper bound is specified in ``metric_type_bounds``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"load_df() expects a pandas DataFrame, got {type(df).__name__}. "
            "To load from a file, use evaluma.load_csv()."
        )

    if metric_type_bounds is not None and (
        norm_ref_low is not None or norm_ref_high is not None
    ):
        raise ValueError(
            "metric_type_bounds cannot be combined with norm_ref_low or norm_ref_high. "
            "Use metric_type_bounds to specify all bounds, or use norm_ref_low/norm_ref_high."
        )

    df = df.copy()
    rename = {model: "model", dataset: "dataset", metric: "metric", score: "score"}
    if seed is not None:
        rename[seed] = "seed"
    df = df.rename(columns=rename)

    metrics_per_dataset = df.groupby(["model", "dataset"])["metric"].nunique()
    if (metrics_per_dataset > 1).any():
        raise ValueError("more than one metric found per (model, dataset) cell")

    check_df = df.drop_duplicates(subset=["model", "dataset"])
    pivot = check_df.pivot(index="model", columns="dataset", values="score")
    missing = pivot.isna()
    if missing.to_numpy().any():
        missing_cells = [
            (m, d)
            for m in pivot.index
            for d in pivot.columns
            if pd.isna(pivot.loc[m, d])
        ]
        if drop_incomplete:
            complete_models = pivot.index[~missing.any(axis=1)]
            df = df[df["model"].isin(complete_models)].reset_index(drop=True)
        else:
            cell_str = ", ".join(f"({m}, {d})" for m, d in missing_cells)
            raise ValueError(f"Incomplete score matrix: missing cells {cell_str}")

    if seed is not None:
        df = df[["model", "dataset", "metric", "score", "seed"]]
    else:
        df = df[["model", "dataset", "metric", "score"]]

    raw_runs = None
    if seed is not None:
        raw_runs = df[["model", "dataset", "seed", "score"]].reset_index(drop=True)
        raw_matrix = (
            df.groupby(["model", "dataset"], as_index=False)["score"]
            .mean()
            .pivot(index="model", columns="dataset", values="score")
        )
    else:
        raw_matrix = df.pivot(index="model", columns="dataset", values="score")
    raw_matrix.columns.name = None

    if metric_type_bounds is not None:
        dataset_metric_map = (
            df.drop_duplicates("dataset").set_index("dataset")["metric"].to_dict()
        )
        norm_ref_low, norm_ref_high, metric_direction = _resolve_metric_type_bounds(
            metric_type_bounds, dataset_metric_map, raw_matrix, metric_direction
        )

    return Benchmark(
        raw_matrix,
        norm_ref_low=norm_ref_low,
        norm_ref_high=norm_ref_high,
        metric_direction=metric_direction,
        raw_runs=raw_runs,
    )


def load_csv(
    path,
    *,
    model="model",
    dataset="dataset",
    metric="metric",
    score="score",
    seed=None,
    metric_type_bounds=None,
    norm_ref_low=None,
    norm_ref_high=None,
    metric_direction=None,
    drop_incomplete=False,
):
    """Load a benchmark CSV file and return a ready-to-use Benchmark object.

    Args:
        path: Path to the CSV file.
        model: Column name for the model identifier.
        dataset: Column name for the dataset identifier.
        metric: Column name for the metric identifier.
        score: Column name for the score values.
        seed: Column name for the random seed.
        metric_type_bounds: See :func:`evaluma.load_df`.
        norm_ref_low: See :func:`evaluma.load_df`.
        norm_ref_high: See :func:`evaluma.load_df`.
        metric_direction: See :func:`evaluma.load_df`.
        drop_incomplete: See :func:`evaluma.load_df`.

    Returns:
        Benchmark: Normalized benchmark ready for analysis.
    """
    df = pd.read_csv(path)
    return load_df(
        df,
        model=model,
        dataset=dataset,
        metric=metric,
        score=score,
        seed=seed,
        metric_type_bounds=metric_type_bounds,
        norm_ref_low=norm_ref_low,
        norm_ref_high=norm_ref_high,
        metric_direction=metric_direction,
        drop_incomplete=drop_incomplete,
    )


def _resolve_metric_type_bounds(
    metric_type_bounds, dataset_metric_map, raw_matrix, metric_direction_override
):
    """Resolve per-dataset normalization bounds and directions from the metric registry.

    For each dataset, consults ``metric_type_bounds`` first, then falls back to the
    built-in registry for metrics with natural bounds. Raises if an unbounded metric
    (rmse, mae, mse) has no entry in ``metric_type_bounds``.

    Args:
        metric_type_bounds: Dict mapping metric names → ``(low, high)`` tuples.
        dataset_metric_map: Dict mapping dataset names → metric name strings.
        raw_matrix: Model × dataset score DataFrame (used to resolve model-name bounds).
        metric_direction_override: Optional dict mapping dataset names → ``"min"``/``"max"``;
            these entries override registry-inferred directions.

    Returns:
        tuple: ``(norm_ref_low, norm_ref_high, metric_direction)`` where the first two are
        ``pd.Series`` keyed by dataset and the last is a dict (or ``None`` if empty).
    """
    from evaluma.metric_registry import get_direction, get_natural_bounds

    mtb_lower = {k.lower(): v for k, v in metric_type_bounds.items()}

    low_dict = {}
    high_dict = {}
    direction_dict = {}

    for dataset, metric_name in dataset_metric_map.items():
        metric_lower = metric_name.lower()

        if metric_lower in mtb_lower:
            low, high = mtb_lower[metric_lower]
            if isinstance(high, str):
                model_name = high
                if model_name not in raw_matrix.index:
                    raise ValueError(
                        f"Reference model '{model_name}' not found in score matrix. "
                        f"Available models: {list(raw_matrix.index)}"
                    )
                high = raw_matrix.loc[model_name, dataset]
            low_dict[dataset] = float(low)
            high_dict[dataset] = float(high)
            direction_dict[dataset] = get_direction(metric_lower)
        else:
            direction = get_direction(metric_lower)  # raises ValueError for unknowns
            nat_low, nat_high = get_natural_bounds(metric_lower)
            if nat_high is None:
                raise ValueError(
                    f"Metric '{metric_name}' on dataset '{dataset}' has no natural "
                    f"upper bound: add '{metric_lower}' to metric_type_bounds in your config."
                )
            low_dict[dataset] = nat_low
            high_dict[dataset] = nat_high
            direction_dict[dataset] = direction

    direction_dict.update(metric_direction_override or {})
    return (
        pd.Series(low_dict),
        pd.Series(high_dict),
        direction_dict or None,
    )
