import os
import sys
import warnings

import click
import yaml

import evaluma


def _parse_metric_direction(ctx, param, value):
    """Parse ``KEY:min`` / ``KEY:max`` tokens into a metric-direction dict.

    Args:
        ctx: Click context (unused; required by the callback protocol).
        param: Click parameter (unused).
        value: Tuple of strings, each formatted as ``"KEY:min"`` or
            ``"KEY:max"``.

    Returns:
        dict | None: Mapping from dataset name to ``"min"`` or ``"max"``,
        or ``None`` when ``value`` is empty.

    Raises:
        click.BadParameter: If a token is malformed or the direction is not
            ``"min"`` or ``"max"``.
    """
    result = {}
    for token in value:
        if ":" not in token:  # pragma: no cover
            raise click.BadParameter(f"Expected KEY:min or KEY:max, got '{token}'")
        key, direction = token.split(":", 1)
        if direction not in ("min", "max"):  # pragma: no cover
            raise click.BadParameter(
                f"Direction must be 'min' or 'max', got '{direction}'"
            )
        result[key] = direction
    return result or None


def _common_options(f):
    """Attach shared CLI options to a Click command."""
    f = click.argument("csv_path")(f)
    f = click.option("--model", default="model")(f)
    f = click.option("--dataset", default="dataset")(f)
    f = click.option("--metric", default="metric")(f)
    f = click.option("--score", default="score")(f)
    f = click.option("--config", "config_path", default=None, help="YAML config file")(
        f
    )
    f = click.option(
        "--metric-direction",
        multiple=True,
        callback=_parse_metric_direction,
        is_eager=False,
        help="KEY:min or KEY:max (repeatable)",
    )(f)
    f = click.option("--output", "output_dir", default=".", show_default=True)(f)
    return f


def _load_bench(
    csv_path,
    model,
    dataset,
    metric,
    score,
    config_path,
    metric_direction,
    output_dir,
    seed=None,
):
    """Load a CSV and return a normalized Benchmark, merging CLI args with config.

    Args:
        csv_path: Path to the input CSV file.
        model: CLI value for the model column name.
        dataset: CLI value for the dataset column name.
        metric: CLI value for the metric column name.
        score: CLI value for the score column name.
        config_path: Optional path to a YAML config file.
        metric_direction: Parsed metric-direction dict (or ``None``).
        output_dir: Path to the output directory (created if absent).
        seed: Optional column name for the random seed.

    Returns:
        Benchmark: Loaded and normalized benchmark.
    """
    cfg = {}
    if config_path:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}

    col_model = model if model != "model" else cfg.get("model", model)
    col_dataset = dataset if dataset != "dataset" else cfg.get("dataset", dataset)
    col_metric = metric if metric != "metric" else cfg.get("metric", metric)
    col_score = score if score != "score" else cfg.get("score", score)
    md = metric_direction or cfg.get("metric_direction") or None

    raw_mtb = cfg.get("metric_type_bounds")
    if raw_mtb is not None:
        mtb = {k: tuple(v) for k, v in raw_mtb.items()}
    else:
        mtb = None

    os.makedirs(output_dir, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if mtb is not None:
            bench = evaluma.load_csv(
                csv_path,
                model=col_model,
                dataset=col_dataset,
                metric=col_metric,
                score=col_score,
                seed=seed,
                metric_direction=md,
                metric_type_bounds=mtb,
            )
        else:
            bench = evaluma.load_csv(
                csv_path,
                model=col_model,
                dataset=col_dataset,
                metric=col_metric,
                score=col_score,
                seed=seed,
                metric_direction=md,
                norm_ref_low=0.0,
                norm_ref_high=1.0,
            )
    return bench


def _save(result, stem, output_dir):
    """Serialize a result to CSV and PNG inside ``output_dir``."""
    csv_path = os.path.join(output_dir, f"{stem}.csv")
    png_path = os.path.join(output_dir, f"{stem}.png")
    result.table.to_csv(csv_path, index=False)
    fig = result.plot()
    fig.savefig(png_path, bbox_inches="tight")


@click.group()
def main():
    """evaluma — ML benchmark evaluation tools."""


@main.command()
@_common_options
def report(
    csv_path, model, dataset, metric, score, config_path, metric_direction, output_dir
):
    """Run all three analyses and write results to ``--output``."""
    bench = _load_bench(
        csv_path,
        model,
        dataset,
        metric,
        score,
        config_path,
        metric_direction,
        output_dir,
    )
    _save(bench.aggregate_ranking(), "aggregate_ranking", output_dir)
    _save(bench.bayesian_comparison(), "bayesian_comparison", output_dir)
    _save(bench.performance_profiles(), "performance_profiles", output_dir)
    _save(bench.frequentist_comparison(), "frequentist_comparison", output_dir)


@main.command()
@_common_options
@click.option("--seed", default=None, help="Column name for the random seed.")
def rank(
    csv_path,
    model,
    dataset,
    metric,
    score,
    config_path,
    metric_direction,
    output_dir,
    seed,
):
    """Compute IQM rankings (requires seed column) and write iqm_ranking.{csv,png}."""
    bench = _load_bench(
        csv_path,
        model,
        dataset,
        metric,
        score,
        config_path,
        metric_direction,
        output_dir,
        seed=seed,
    )
    if bench._raw_runs is None:
        click.echo(
            "Error: iqm_ranking() requires multiple seeds — "
            "use evaluma aggregate for single-run data.",
            err=True,
        )
        sys.exit(1)
    _save(bench.iqm_ranking(), "iqm_ranking", output_dir)


@main.command()
@_common_options
@click.option(
    "--agg",
    default="trimmed_mean",
    show_default=True,
    help="Aggregation mode: trimmed_mean, mean, or median.",
)
def aggregate(
    csv_path,
    model,
    dataset,
    metric,
    score,
    config_path,
    metric_direction,
    output_dir,
    agg,
):
    """Compute point-estimate aggregate ranking and write aggregate_ranking.csv/png."""
    bench = _load_bench(
        csv_path,
        model,
        dataset,
        metric,
        score,
        config_path,
        metric_direction,
        output_dir,
    )
    _save(bench.aggregate_ranking(agg=agg), "aggregate_ranking", output_dir)


@main.command()
@_common_options
def compare(
    csv_path, model, dataset, metric, score, config_path, metric_direction, output_dir
):
    """Compute Bayesian pairwise comparisons and write results."""
    bench = _load_bench(
        csv_path,
        model,
        dataset,
        metric,
        score,
        config_path,
        metric_direction,
        output_dir,
    )
    _save(bench.bayesian_comparison(), "bayesian_comparison", output_dir)


@main.command()
@_common_options
@click.option(
    "--reference",
    default=None,
    help="Reference model for one-vs-all comparison.",
)
@click.option(
    "--alpha",
    default=0.05,
    show_default=True,
    type=float,
    help="Significance level for the significant column.",
)
def frequentist(
    csv_path,
    model,
    dataset,
    metric,
    score,
    config_path,
    metric_direction,
    output_dir,
    reference,
    alpha,
):
    """Compute Friedman + Nemenyi (all-pairs) or Wilcoxon + Holm (reference) comparison."""
    bench = _load_bench(
        csv_path,
        model,
        dataset,
        metric,
        score,
        config_path,
        metric_direction,
        output_dir,
    )
    _save(
        bench.frequentist_comparison(reference=reference, alpha=alpha),
        "frequentist_comparison",
        output_dir,
    )


@main.command()
@_common_options
def profiles(
    csv_path, model, dataset, metric, score, config_path, metric_direction, output_dir
):
    """Compute Dolan-Moré performance profiles and write results."""
    bench = _load_bench(
        csv_path,
        model,
        dataset,
        metric,
        score,
        config_path,
        metric_direction,
        output_dir,
    )
    _save(bench.performance_profiles(), "performance_profiles", output_dir)
