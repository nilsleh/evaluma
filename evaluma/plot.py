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


def plot_cd_diagram(
    avg_ranks: pd.Series,
    cd: float,
    *,
    title=None,
    figsize=None,
):
    """Render a Critical Difference diagram (Demšar 2006).

    Models are placed on a horizontal axis by average rank (rank 1 = best on the
    left). Thick horizontal bars connect cliques of models whose rank gap does not
    exceed the Nemenyi CD scalar. A CD bracket in the top-right corner shows the
    critical difference visually.

    Args:
        avg_ranks: Series mapping model names to average rank (lower = better),
            as produced by :func:`~evaluma.methods.frequentist.compute_frequentist`.
        cd: Nemenyi critical difference scalar.
        title: Optional axes title.
        figsize: Figure size ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    models_sorted = avg_ranks.sort_values().index.tolist()
    n = len(models_sorted)
    ranks_sorted = avg_ranks[models_sorted]

    # Greedy forward scan clique detection (autorank style)
    groups = []
    cur_max_j = -1
    for i in range(n):
        max_j = None
        for j in range(i + 1, n):
            if ranks_sorted.iloc[j] - ranks_sorted.iloc[i] <= cd:
                max_j = j
        if max_j is not None and max_j > cur_max_j:
            cur_max_j = max_j
            groups.append((i, max_j))

    n_bars = len(groups)
    fig_w = max(6.0, n * 1.5)
    fig_h = max(3.0, 2.0 + n_bars * 0.4)
    fig, ax = plt.subplots(figsize=figsize or (fig_w, fig_h))

    rank_vals = avg_ranks.values
    r_min_data = float(rank_vals.min())
    r_max_data = float(rank_vals.max())

    # CD bracket anchored at the right end; expand xlim to fit it
    cd_right = r_max_data + 0.35
    cd_left = cd_right - cd
    r_left = min(r_min_data - 0.5, cd_left - 0.1)
    r_right = max(r_max_data + 0.5, cd_right + 0.1)
    ax.set_xlim(r_left, r_right)

    ax.axhline(0.0, color="black", lw=1.5, zorder=1)

    for i, model in enumerate(models_sorted):
        r = float(avg_ranks[model])
        ax.plot(r, 0.0, "o", color="black", ms=5, zorder=3)
        if i % 2 == 0:
            y_label = -0.15
        else:
            y_label = -0.50
            ax.plot([r, r], [-0.05, -0.43], color="black", lw=0.8, zorder=1)
        ax.text(
            r,
            y_label,
            f"{model}\n({r:.2f})",
            ha="right",
            va="top",
            fontsize=9,
            rotation=30,
        )

    for bar_i, (start, end) in enumerate(groups):
        r0 = float(avg_ranks[models_sorted[start]])
        r1 = float(avg_ranks[models_sorted[end]])
        y_bar = 0.35 + bar_i * 0.3
        ax.plot(
            [r0, r1],
            [y_bar, y_bar],
            color="black",
            lw=4,
            solid_capstyle="butt",
            zorder=2,
        )

    y_top = (0.35 + n_bars * 0.3 + 0.2) if n_bars else 0.5

    # CD bracket in top-right corner
    y_bracket = y_top + 0.12
    cd_tick_h = 0.06
    ax.plot([cd_left, cd_right], [y_bracket, y_bracket], color="black", lw=1.5)
    ax.plot(
        [cd_left, cd_left],
        [y_bracket - cd_tick_h, y_bracket + cd_tick_h],
        color="black",
        lw=1.5,
    )
    ax.plot(
        [cd_right, cd_right],
        [y_bracket - cd_tick_h, y_bracket + cd_tick_h],
        color="black",
        lw=1.5,
    )
    ax.text(
        (cd_left + cd_right) / 2,
        y_bracket + cd_tick_h + 0.02,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xlabel("Average rank (lower is better)")
    ax.set_ylim(-1.5, y_bracket + 0.25)
    ax.yaxis.set_visible(False)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    if title:  # pragma: no cover
        ax.set_title(title)

    return fig


def plot_frequentist_reference_bars(
    table: pd.DataFrame,
    reference: str,
    alpha: float,
    *,
    title=None,
    figsize=None,
):
    """Render frequentist reference-mode results as horizontal bars.

    Each bar shows the Holm-corrected p-value for a model vs the reference.
    A vertical dashed line marks the significance threshold.

    Args:
        table: DataFrame with columns ``model_a``, ``model_b``,
            ``p_value_corrected``, ``significant``, as produced by
            :func:`~evaluma.methods.frequentist.compute_frequentist` in
            reference mode.
        reference: Name of the reference model.
        alpha: Significance threshold; used to position the dashed line.
        title: Optional figure title.
        figsize: Figure size ``(width, height)`` in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    df = table[table["model_a"] == reference].copy()
    df = df.sort_values("p_value_corrected", ascending=True)

    models = df["model_b"].tolist()
    p_corr = df["p_value_corrected"].values
    colors = ["#DC2626" if sig else "#9CA3AF" for sig in df["significant"]]

    n = len(models)
    fig, ax = plt.subplots(figsize=figsize or (8, max(3, n * 0.5)))

    ax.barh(models, p_corr, color=colors)
    ax.axvline(x=alpha, color="black", linestyle="--", lw=1.2)
    ax.set_xlabel("Holm-corrected p-value")
    ax.set_xlim(0, max(1.0, float(p_corr.max()) * 1.05))

    if title:  # pragma: no cover
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
