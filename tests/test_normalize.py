import warnings

import numpy as np
import pytest

from evaluma.normalize import normalize


def _make_matrix(score_df):
    return score_df.pivot(index="model", columns="dataset", values="score")


def test_observed_min_maps_to_zero(score_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = normalize(_make_matrix(score_df))
    assert np.allclose(result.min().values, 0.0)


def test_observed_max_maps_to_one(score_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = normalize(_make_matrix(score_df))
    assert np.allclose(result.max().values, 1.0)


def test_observed_bounds_emits_warning(score_df):
    with pytest.warns(UserWarning, match="Normalization bounds depend"):
        normalize(_make_matrix(score_df))


def test_norm_ref_low_model(score_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = normalize(_make_matrix(score_df), norm_ref_low="C")
    assert (result.loc["C"] <= 0.0).all()
    assert (result.loc["A"] > 0).all()


def test_norm_ref_high_model(score_df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = normalize(_make_matrix(score_df), norm_ref_high="A")
    assert (result.loc["A"] >= 1.0).all()


def test_scalar_dict_bounds(score_df):
    datasets = ["d1", "d2", "d3", "d4"]
    low_dict = {d: 0.0 for d in datasets}
    high_dict = {d: 1.0 for d in datasets}
    result = normalize(
        _make_matrix(score_df), norm_ref_low=low_dict, norm_ref_high=high_dict
    )
    assert ((result >= 0.0) & (result <= 1.0)).all().all()


def test_missing_reference_model_raises(score_df):
    with pytest.raises(ValueError, match="not found"):
        normalize(_make_matrix(score_df), norm_ref_low="nonexistent", norm_ref_high="A")


def test_lower_is_better_inverted(score_df_lowerisbetter):
    # metric_direction keys are DATASET names, not metric names
    datasets = ["d1", "d2", "d3", "d4"]
    matrix = _make_matrix(score_df_lowerisbetter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = normalize(matrix, metric_direction={d: "min" for d in datasets})
    # C has the lowest raw RMSE values → should have the highest normalized score
    assert result.loc["C"].mean() > result.loc["A"].mean()


def test_mixed_direction_with_scalar_bounds():
    # Regression test: only the "min" column should be inverted, not the whole matrix.
    # Model A is best on acc (0.9) and has worst RMSE (0.9); model C is worst on acc
    # (0.2) and has best RMSE (0.2). With correct per-column inversion and [0,1] bounds:
    #   acc column: normalized as-is → A=0.9, C=0.2
    #   rmse column: inverted → A becomes 0.1, C becomes 0.8
    import pandas as pd

    matrix = pd.DataFrame(
        {"acc": {"A": 0.9, "B": 0.5, "C": 0.2}, "rmse": {"A": 0.9, "B": 0.5, "C": 0.2}}
    )
    result = normalize(
        matrix,
        norm_ref_low=0.0,
        norm_ref_high=1.0,
        metric_direction={"rmse": "min"},
    )
    # acc column must NOT be inverted
    assert np.isclose(result.loc["A", "acc"], 0.9)
    assert np.isclose(result.loc["C", "acc"], 0.2)
    # rmse column must BE inverted: low raw RMSE → high normalized score
    assert np.isclose(result.loc["A", "rmse"], 0.1)
    assert np.isclose(result.loc["C", "rmse"], 0.8)
    # all values must be in [0, 1]
    assert ((result >= 0.0) & (result <= 1.0)).all().all()
