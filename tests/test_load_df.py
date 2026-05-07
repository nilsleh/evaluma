import pandas as pd
import pytest

import evaluma


def test_load_df_basic(score_df):
    result = evaluma.load_df(
        score_df, model="model", dataset="dataset", metric="metric", score="score",
        norm_ref_low=0.0, norm_ref_high=1.0,
    )
    from evaluma.benchmark import Benchmark
    assert isinstance(result, Benchmark)
    assert result.scores_.shape == (3, 4)


def test_load_df_custom_column_names(score_df):
    df = score_df.rename(
        columns={"model": "exp", "dataset": "task", "metric": "m", "score": "val"}
    )
    result = evaluma.load_df(
        df, model="exp", dataset="task", metric="m", score="val",
        norm_ref_low=0.0, norm_ref_high=1.0,
    )
    assert result.scores_.shape == (3, 4)


def test_load_df_seed_keeps_raw_rows(score_df_seeded):
    bench = evaluma.load_df(
        score_df_seeded,
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        seed="seed",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    assert bench._raw_runs is not None
    assert len(bench._raw_runs) == 36
    assert "seed" in bench._raw_runs.columns


def test_load_df_seed_column_values_preserved(score_df_seeded):
    bench = evaluma.load_df(
        score_df_seeded,
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        seed="seed",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    assert sorted(bench._raw_runs["seed"].unique()) == [1, 2, 3]
    for _, row in bench._raw_runs.iterrows():
        orig = score_df_seeded[
            (score_df_seeded["model"] == row["model"])
            & (score_df_seeded["dataset"] == row["dataset"])
            & (score_df_seeded["seed"] == row["seed"])
        ]["score"].iloc[0]
        assert abs(row["score"] - orig) < 1e-9


def test_load_df_no_seed_raw_runs_none(score_df):
    bench = evaluma.load_df(
        score_df, model="model", dataset="dataset", metric="metric", score="score",
        norm_ref_low=0.0, norm_ref_high=1.0,
    )
    assert bench._raw_runs is None


def test_load_df_missing_cell_raises(score_df_missing):
    with pytest.raises(ValueError, match="Incomplete score matrix"):
        evaluma.load_df(
            score_df_missing,
            model="model", dataset="dataset", metric="metric", score="score",
            norm_ref_low=0.0, norm_ref_high=1.0,
        )


def test_load_df_multi_metric_raises(score_df):
    df = score_df.copy()
    extra = df[(df["model"] == "A") & (df["dataset"] == "d1")].copy()
    extra["metric"] = "other_metric"
    df = pd.concat([df, extra], ignore_index=True)
    with pytest.raises(ValueError, match="more than one.*metric"):
        evaluma.load_df(
            df, model="model", dataset="dataset", metric="metric", score="score",
            norm_ref_low=0.0, norm_ref_high=1.0,
        )


def test_load_df_type_error_on_path(tmp_path, score_df):
    path = tmp_path / "scores.csv"
    score_df.to_csv(path, index=False)
    with pytest.raises(TypeError, match="load_csv"):
        evaluma.load_df(path)


def test_load_df_type_error_on_string():
    with pytest.raises(TypeError, match="load_csv"):
        evaluma.load_df("results.csv")
