import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

EXPECTED_IQM = {"A": 0.75, "B": 0.5, "C": 0.275}


def test_iqm_raises_without_seeds(bench):
    with pytest.raises(ValueError, match="multiple seeds"):
        bench.iqm_ranking()


def test_iqm_exact(bench_runs):
    result = bench_runs.iqm_ranking(random_state=0)
    computed = result.table.set_index("model")["IQM"].to_dict()
    for model, expected in EXPECTED_IQM.items():
        assert abs(computed[model] - expected) < 1e-6, (
            f"{model}: {computed[model]} != {expected}"
        )


def test_iqm_table_schema(bench_runs):
    result = bench_runs.iqm_ranking(random_state=0)
    assert result.table.columns.tolist() == ["model", "IQM", "CI_low", "CI_high"]


def test_iqm_ci_structure(bench_runs):
    result = bench_runs.iqm_ranking(random_state=0)
    assert (result.table["CI_low"] < result.table["IQM"]).all()
    assert (result.table["IQM"] < result.table["CI_high"]).all()


def test_iqm_ci_width_positive(bench_runs):
    result = bench_runs.iqm_ranking(random_state=0)
    assert (result.table["CI_high"] - result.table["CI_low"] > 0).all()


def test_iqm_ranking_order(bench_runs):
    result = bench_runs.iqm_ranking(random_state=0)
    tbl = result.table.set_index("model")
    assert tbl.loc["A", "IQM"] > tbl.loc["B", "IQM"] > tbl.loc["C", "IQM"]


def test_iqm_plot_returns_figure(bench_runs):
    fig = bench_runs.iqm_ranking(random_state=0).plot()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_iqm_plot_no_show(bench_runs, monkeypatch):
    monkeypatch.setattr(
        plt, "show", lambda: (_ for _ in ()).throw(AssertionError("plt.show called"))
    )
    fig = bench_runs.iqm_ranking(random_state=0).plot()
    plt.close(fig)


def test_iqm_plot_ax_argument(bench_runs):
    existing_fig, existing_ax = plt.subplots()
    fig = bench_runs.iqm_ranking(random_state=0).plot(ax=existing_ax)
    assert fig is existing_ax.get_figure()
    plt.close(fig)
