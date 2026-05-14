import pandas as pd
import yaml
from click.testing import CliRunner

from evaluma.cli import main

_DATASETS = ["d1", "d2", "d3", "d4", "d5"]

SCORES_ROWS = [
    {"model": m, "dataset": d, "metric": "acc", "score": s}
    for m, scores in [
        ("A", [0.9, 0.8, 0.7, 0.6, 0.85]),
        ("B", [0.5, 0.6, 0.4, 0.5, 0.55]),
        ("C", [0.2, 0.3, 0.25, 0.35, 0.22]),
    ]
    for d, s in zip(_DATASETS, scores)
]

SEEDED_ROWS = [
    {"model": m, "dataset": d, "metric": "acc", "score": s + offset, "seed": seed}
    for m, scores in [
        ("A", [0.9, 0.8, 0.7, 0.6, 0.85]),
        ("B", [0.5, 0.6, 0.4, 0.5, 0.55]),
        ("C", [0.2, 0.3, 0.25, 0.35, 0.22]),
    ]
    for d, s in zip(_DATASETS, scores)
    for seed, offset in enumerate([-0.05, 0.0, 0.05], 1)
]


def _write_csv(tmp_path):
    path = tmp_path / "scores.csv"
    pd.DataFrame(SCORES_ROWS).to_csv(path, index=False)
    return str(path)


def _write_seeded_csv(tmp_path):
    path = tmp_path / "seeded.csv"
    pd.DataFrame(SEEDED_ROWS).to_csv(path, index=False)
    return str(path)


def test_report_writes_six_files(tmp_path):
    runner = CliRunner()
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main,
        [
            "report",
            csv,
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--metric",
            "metric",
            "--score",
            "score",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    for name in [
        "aggregate_ranking.csv",
        "aggregate_ranking.png",
        "bayesian_comparison.csv",
        "bayesian_comparison.png",
        "performance_profiles.csv",
        "performance_profiles.png",
        "frequentist_comparison.csv",
        "frequentist_comparison.png",
    ]:
        assert (out / name).exists(), f"Missing: {name}"


def test_report_config_yaml(tmp_path):
    runner = CliRunner()
    csv = _write_csv(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.dump(
            {
                "model": "model",
                "dataset": "dataset",
                "metric": "metric",
                "score": "score",
            }
        )
    )
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main, ["report", csv, "--config", str(config), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (out / "aggregate_ranking.csv").exists()


def test_rank_exits_without_seed(tmp_path):
    runner = CliRunner()
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main,
        [
            "rank",
            csv,
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--metric",
            "metric",
            "--score",
            "score",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "aggregate" in result.output.lower()


def test_rank_succeeds_with_seed(tmp_path):
    runner = CliRunner()
    csv = _write_seeded_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main,
        [
            "rank",
            csv,
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--metric",
            "metric",
            "--score",
            "score",
            "--seed",
            "seed",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "iqm_ranking.csv").exists()


def test_metric_direction_cli_syntax(tmp_path):
    runner = CliRunner()
    csv = _write_seeded_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main,
        [
            "rank",
            csv,
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--metric",
            "metric",
            "--score",
            "score",
            "--seed",
            "seed",
            "--metric-direction",
            "acc:max",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output


def test_rank_subcommand(tmp_path):
    runner = CliRunner()
    csv = _write_seeded_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main,
        [
            "rank",
            csv,
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--metric",
            "metric",
            "--score",
            "score",
            "--seed",
            "seed",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "iqm_ranking.csv").exists()
    assert (out / "iqm_ranking.png").exists()


def test_aggregate_subcommand_creates_files(tmp_path):
    runner = CliRunner()
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main,
        [
            "aggregate",
            csv,
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--metric",
            "metric",
            "--score",
            "score",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "aggregate_ranking.csv").exists()
    assert (out / "aggregate_ranking.png").exists()


def test_aggregate_subcommand_agg_option(tmp_path):
    runner = CliRunner()
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main,
        [
            "aggregate",
            csv,
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--metric",
            "metric",
            "--score",
            "score",
            "--agg",
            "mean",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output


def test_compare_subcommand(tmp_path):
    runner = CliRunner()
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main,
        [
            "compare",
            csv,
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--metric",
            "metric",
            "--score",
            "score",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "bayesian_comparison.csv").exists()
    assert (out / "bayesian_comparison.png").exists()


def test_profiles_subcommand(tmp_path):
    runner = CliRunner()
    csv = _write_csv(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main,
        [
            "profiles",
            csv,
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--metric",
            "metric",
            "--score",
            "score",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "performance_profiles.csv").exists()
    assert (out / "performance_profiles.png").exists()


def _write_mixed_csv(tmp_path):
    """CSV with IoU and RMSE datasets; 'baseline' model present for RMSE ceiling."""
    rows = []
    for m, iou_scores, rmse_scores in [
        ("A", [0.9, 0.8], [1.5, 2.0]),
        ("B", [0.6, 0.7], [2.5, 3.0]),
        ("baseline", [0.4, 0.5], [4.0, 5.0]),
    ]:
        for i, s in enumerate(iou_scores):
            rows.append(
                {"model": m, "dataset": f"d_iou_{i}", "metric": "iou", "score": s}
            )
        for i, s in enumerate(rmse_scores):
            rows.append(
                {"model": m, "dataset": f"d_rmse_{i}", "metric": "rmse", "score": s}
            )
    path = tmp_path / "mixed.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_cli_reads_metric_type_bounds_from_yaml(tmp_path):
    """YAML config with metric_type_bounds produces a successful aggregate."""
    runner = CliRunner()
    csv = _write_mixed_csv(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.dump(
            {
                "metric_type_bounds": {
                    "rmse": [0.0, "baseline"],
                }
            }
        )
    )
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main, ["aggregate", csv, "--config", str(config), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (out / "aggregate_ranking.csv").exists()


def test_cli_no_metric_type_bounds_keeps_legacy_bounds(tmp_path):
    """Without metric_type_bounds in config, the CLI uses the legacy 0/1 bounds."""
    runner = CliRunner()
    csv = _write_csv(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.dump({"model": "model", "dataset": "dataset"}))
    out = tmp_path / "out"
    out.mkdir()
    result = runner.invoke(
        main, ["aggregate", csv, "--config", str(config), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (out / "aggregate_ranking.csv").exists()
