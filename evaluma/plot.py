import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd


def plot_aggregate_ranking(
    table: pd.DataFrame, *, figsize=None, model_colors=None, title=None, ax=None
):
    """Render aggregate scores as a horizontal bar chart (no CI whiskers).

    Args:
        table: DataFrame with columns ``model`` and ``score``.
        figsize: Figure size ``(width, height)`` in inches.
        model_colors: List of colors, one per model in row order.
        title: Optional axes title.
        ax: Existing axes to draw into; a new figure is created if ``None``.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6, 3))
    else:
        fig = ax.get_figure()

    models = table["model"].tolist()
    scores = table["score"].values
    colors = model_colors or [f"C{i}" for i in range(len(models))]
    y_pos = np.arange(len(models))
    ax.barh(y_pos, scores, color=colors, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Score")
    if title:  # pragma: no cover
        ax.set_title(title)
    return fig


def plot_iqm_ranking(
    table: pd.DataFrame, *, figsize=None, model_colors=None, title=None, ax=None
):
    """Render IQM scores as a horizontal bar chart with CI error bars.

    Args:
        table: DataFrame with columns ``model``, ``IQM``, ``CI_low``,
            ``CI_high`` as produced by
            :func:`~evaluma.methods.iqm.compute_iqm`.
        figsize: Figure size ``(width, height)`` in inches.
        model_colors: List of colors, one per model in row order.
        title: Optional axes title.
        ax: Existing axes to draw into; a new figure is created if
            ``None``.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6, 3))
    else:
        fig = ax.get_figure()

    models = table["model"].tolist()
    iqms = table["IQM"].values
    has_ci = not table["CI_low"].isna().all()

    xerr = None
    if has_ci:
        ci_low = table["CI_low"].values
        ci_high = table["CI_high"].values
        xerr = np.vstack([iqms - ci_low, ci_high - iqms])

    colors = model_colors or [f"C{i}" for i in range(len(models))]
    y_pos = np.arange(len(models))
    ax.barh(y_pos, iqms, xerr=xerr, color=colors, align="center", capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("IQM")
    if title:  # pragma: no cover
        ax.set_title(title)
    return fig


def plot_bayesian_heatmap(table: pd.DataFrame, *, title=None, figsize=None, **_kwargs):
    """Render Bayesian pairwise probabilities as a matplotlib heatmap.

    Each cell ``(i, j)`` shows ``P(model_i > model_j)``.

    Args:
        table: DataFrame with columns ``model_a``, ``model_b``,
            ``p_a_better``, ``p_equiv``, ``p_b_better``.
        title: Optional figure title.
        figsize: Figure size ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    models = sorted(set(table["model_a"]) | set(table["model_b"]))
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}
    matrix = np.full((n, n), np.nan)
    for _, row in table.iterrows():
        i, j = idx[row["model_a"]], idx[row["model_b"]]
        matrix[i, j] = row["p_a_better"]
        matrix[j, i] = row["p_b_better"]

    size = figsize or (max(6, n * 2), max(5, n * 1.8))
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="coolwarm", aspect="equal")

    fs = max(8, (size[1] / n) * 9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(row > col)", fontsize=int(fs * 1.4))
    cbar.ax.tick_params(labelsize=fs)

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=fs * 1.1,
                )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=int(fs * 1.25))
    ax.set_yticklabels(models, fontsize=int(fs * 1.25))
    ax.set_xlabel("model_b", fontsize=int(fs * 1.4))
    ax.set_ylabel("model_a", fontsize=int(fs * 1.4))
    if title:
        ax.set_title(title)
    return fig


def plot_bayesian_reference_bars(
    table: pd.DataFrame, reference: str, *, title=None, figsize=None
):
    """Render Bayesian comparison against a reference as stacked horizontal bars.

    Each bar represents one model compared to the reference. Blue = P(model >
    reference), grey = P(equivalent), red = P(reference > model). Bars are
    sorted by P(model > reference) descending.

    Args:
        table: DataFrame with columns ``model_a``, ``model_b``,
            ``p_a_better``, ``p_equiv``, ``p_b_better``. Expects
            ``model_a == reference`` for all rows (as produced by
            :func:`~evaluma.methods.bayesian.compute_bayesian` in reference
            mode).
        reference: Name of the reference model.
        title: Optional figure title.
        figsize: Figure size ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    df = table[table["model_a"] == reference].copy()
    df = df.sort_values("p_b_better", ascending=True)

    models = df["model_b"].tolist()
    p_model = df["p_b_better"].values
    p_equiv = df["p_equiv"].values
    p_ref = df["p_a_better"].values

    n = len(models)
    fig, ax = plt.subplots(figsize=figsize or (8, max(3, n * 0.5)))

    ax.barh(models, p_model, color="#2563EB", label=f"P(model > {reference})")
    ax.barh(models, p_equiv, left=p_model, color="#9CA3AF", label="P(equivalent)")
    ax.barh(
        models,
        p_ref,
        left=p_model + p_equiv,
        color="#DC2626",
        label=f"P({reference} > model)",
    )

    for i, (pw, pe, pr) in enumerate(zip(p_model, p_equiv, p_ref)):
        if pw > 0.06:
            ax.text(
                pw / 2,
                i,
                f"{pw:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        if pe > 0.06:
            ax.text(
                pw + pe / 2,
                i,
                f"{pe:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        if pr > 0.06:
            ax.text(
                pw + pe + pr / 2,
                i,
                f"{pr:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    if title:
        ax.set_title(title)
    return fig


def plot_performance_profiles(
    table: pd.DataFrame, *, figsize=None, model_colors=None, title=None, ax=None
):
    """Render Dolan-Moré performance profile curves.

    The x-axis uses a native log₁₀ scale with raw τ ratio values (1, 2, 5, 10…),
    following ML-GYM (Batra et al., 2025) and the AutoML Decathlon (Roberts et al.,
    2022). τ = 1 means tied for best; τ = 10 means 10× worse than the best model.

    Args:
        table: Long-format DataFrame with columns ``tau``, ``model``,
            ``fraction_within_tau``.
        figsize: Figure size in inches.
        model_colors: Dict mapping model names to colors, or a list in
            model order.
        title: Optional axes title.
        ax: Existing axes to draw into; a new figure is created if
            ``None``.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    models = table["model"].unique().tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (6, 4))
    else:  # pragma: no cover
        fig = ax.get_figure()

    colors = model_colors or {m: f"C{i}" for i, m in enumerate(models)}
    for model in models:
        sub = table[table["model"] == model].sort_values("tau")
        color = colors[model] if isinstance(colors, dict) else None
        ax.plot(
            sub["tau"].values,
            sub["fraction_within_tau"].values,
            label=model,
            color=color,
            drawstyle="steps-post",
        )

    ax.set_xscale("log", base=10)
    _fmt = matplotlib.ticker.ScalarFormatter()
    _fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(_fmt)
    ax.set_xlabel("τ (performance ratio)")
    ax.set_ylabel("Fraction of datasets within τ")
    ax.set_xlim(left=1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    if title:  # pragma: no cover
        ax.set_title(title)
    return fig
