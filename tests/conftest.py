import pandas as pd
import pytest


@pytest.fixture
def score_df():
    models = ["A", "B", "C"]
    datasets = ["d1", "d2", "d3", "d4"]
    base = {
        "A": [0.9, 0.8, 0.7, 0.6],
        "B": [0.5, 0.6, 0.4, 0.5],
        "C": [0.2, 0.3, 0.25, 0.35],
    }
    rows = []
    for model in models:
        for i, dataset in enumerate(datasets):
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "metric": "acc",
                    "score": base[model][i],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def score_df_seeded():
    models = ["A", "B", "C"]
    datasets = ["d1", "d2", "d3", "d4"]
    base = {
        "A": [0.9, 0.8, 0.7, 0.6],
        "B": [0.5, 0.6, 0.4, 0.5],
        "C": [0.2, 0.3, 0.25, 0.35],
    }
    offsets = [-0.05, 0.0, 0.05]
    rows = []
    for model in models:
        for i, dataset in enumerate(datasets):
            for seed, offset in enumerate(offsets, 1):
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "metric": "acc",
                        "score": base[model][i] + offset,
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def score_df_missing(score_df):
    mask = ~((score_df["model"] == "C") & (score_df["dataset"] == "d4"))
    return score_df[mask].reset_index(drop=True)


@pytest.fixture
def score_df_lowerisbetter(score_df):
    df = score_df.copy()
    df["metric"] = "rmse"
    return df


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path


@pytest.fixture
def bench_runs(score_df_seeded):
    import evaluma

    return evaluma.load_df(
        score_df_seeded,
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        seed="seed",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )


@pytest.fixture
def bench(score_df):
    import evaluma

    return evaluma.load_df(
        score_df,
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
