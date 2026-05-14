import numpy as np
import pandas as pd


class AggregateResult:
    """Result of :meth:`~evaluma.benchmark.Benchmark.aggregate_ranking`."""

    def __init__(self, table: pd.DataFrame):
        """Args:
        table: DataFrame with columns ``model`` and ``score``.
        """
        self.table = table

    def plot(self, figsize=None, model_colors=None, title=None, ax=None):
        """Render a horizontal bar chart of aggregate scores.

        Args:
            figsize: Figure size ``(width, height)`` in inches.
            model_colors: List of colors, one per model in table order.
            title: Optional axes title.
            ax: Existing axes to draw into; a new figure is created if
                ``None``.

        Returns:
            matplotlib.figure.Figure: The rendered figure.
        """
        from evaluma.plot import plot_aggregate_ranking

        return plot_aggregate_ranking(
            self.table, figsize=figsize, model_colors=model_colors, title=title, ax=ax
        )


class IQMResult:
    """Result of :meth:`~evaluma.benchmark.Benchmark.iqm_ranking`."""

    def __init__(self, table: pd.DataFrame):
        """Args:
        table: DataFrame with columns ``model``, ``IQM``, ``CI_low``,
            ``CI_high``.
        """
        self.table = table

    def plot(self, figsize=None, model_colors=None, title=None, ax=None):
        """Render a horizontal bar chart of IQM scores with CI error bars.

        Args:
            figsize: Matplotlib figure size ``(width, height)`` in inches.
            model_colors: List of colors, one per model in table order.
            title: Optional axes title.
            ax: Existing axes to draw into; a new figure is created if
                ``None``.

        Returns:
            matplotlib.figure.Figure: The rendered figure.

        Example:
            >>> result = bench.iqm_ranking()
            >>> fig = result.plot(figsize=(8, 4))
        """
        from evaluma.plot import plot_iqm_ranking

        return plot_iqm_ranking(
            self.table, figsize=figsize, model_colors=model_colors, title=title, ax=ax
        )


class BayesianResult:
    """Result of :meth:`~evaluma.benchmark.Benchmark.bayesian_comparison`."""

    def __init__(self, table: pd.DataFrame, reference=None):
        """Args:
        table: DataFrame with columns ``model_a``, ``model_b``,
            ``p_a_better``, ``p_equiv``, ``p_b_better``.
        reference: Reference model name if the comparison was run in
            reference mode; ``None`` for all-pairs mode.
        """
        self.table = table
        self.reference = reference

    def plot(self, title=None):
        """Render the comparison result.

        In all-pairs mode renders a pairwise heatmap. In reference mode
        renders a stacked horizontal bar chart sorted by P(model > reference).

        Args:
            title: Optional figure title.

        Returns:
            matplotlib.figure.Figure: The rendered figure.

        Example:
            >>> result = bench.bayesian_comparison()
            >>> fig = result.plot()
        """
        if self.reference is not None:
            from evaluma.plot import plot_bayesian_reference_bars

            return plot_bayesian_reference_bars(self.table, self.reference, title=title)
        from evaluma.plot import plot_bayesian_heatmap

        return plot_bayesian_heatmap(self.table, title=title)


class FrequentistResult:
    """Result of :meth:`~evaluma.benchmark.Benchmark.frequentist_comparison`."""

    def __init__(
        self,
        table: pd.DataFrame,
        avg_ranks: pd.Series,
        friedman_statistic: float,
        friedman_p_value: float,
        reference=None,
        alpha=0.05,
        cd=None,
    ):
        """Args:
        table: DataFrame schema depends on mode. All-pairs: ``model_a``,
            ``model_b``, ``rank_diff``, ``p_value``, ``significant``.
            Reference: ``model_a``, ``model_b``, ``w_statistic``, ``p_value``,
            ``p_value_corrected``, ``significant``.
        avg_ranks: Series mapping model name to average rank across datasets
            (rank 1 = best).
        friedman_statistic: Friedman chi-squared test statistic.
        friedman_p_value: Friedman test p-value.
        reference: Reference model name if the comparison was run in reference
            mode; ``None`` for all-pairs mode.
        alpha: Significance level used for the ``significant`` column.
        cd: Nemenyi critical difference scalar; ``None`` in reference mode.
        """
        self.table = table
        self.avg_ranks = avg_ranks
        self.friedman_statistic = friedman_statistic
        self.friedman_p_value = friedman_p_value
        self.reference = reference
        self.alpha = alpha
        self.cd = cd

    def plot(self, title=None):
        """Render the comparison result.

        In all-pairs mode renders a CD diagram (Demšar 2006) with the Nemenyi
        critical difference bracket. In reference mode renders a horizontal bar
        chart of Holm-corrected p-values.

        Args:
            title: Optional figure title.

        Returns:
            matplotlib.figure.Figure: The rendered figure.

        Example:
            >>> result = bench.frequentist_comparison()
            >>> fig = result.plot()
        """
        if self.reference is not None:
            from evaluma.plot import plot_frequentist_reference_bars

            return plot_frequentist_reference_bars(
                self.table, self.reference, self.alpha, title=title
            )
        from evaluma.plot import plot_cd_diagram

        return plot_cd_diagram(self.avg_ranks, self.cd, title=title)


class ProfileResult:
    """Result of :meth:`~evaluma.benchmark.Benchmark.performance_profiles`."""

    def __init__(self, table: pd.DataFrame):
        """Args:
        table: Long-format DataFrame with columns ``tau``, ``model``,
            ``fraction_within_tau``. ``tau`` holds raw ratio values ≥ 1;
            the plot renders on a native log₁₀(τ) axis. Use ``.aup`` for
            the scalar Area Under the Profile summary.
        """
        self.table = table

    @property
    def aup(self) -> pd.Series:
        """Area Under the Profile in log₁₀(τ) space (left Riemann sum).

        For each model, integrates the step-function profile curve over
        log₁₀(τ) using consecutive tau breakpoints as the grid:

            AUP = Σ (log₁₀(τ_{i+1}) − log₁₀(τ_i)) · ρ(τ_i)

        AUP is unnormalized: its scale depends on τ_max (the worst-case ratio
        in this run). It is meaningful for within-benchmark comparison but not
        across benchmarks with different τ_max values.

        References:
            Roberts et al. (2022). AutoML Decathlon.
            Dahl et al. (2023). AlgoPerf. arXiv:2306.07179.
            Batra et al. (2025). ML-GYM. arXiv:2502.14499.

        Returns:
            pd.Series indexed by model name, values ≥ 0.
        """
        models = self.table["model"].unique()
        aup_values = {}
        for model in models:
            sub = self.table[self.table["model"] == model].sort_values("tau")
            taus = sub["tau"].values
            fracs = sub["fraction_within_tau"].values
            log_taus = np.log10(taus)
            aup_values[model] = float(
                np.sum((log_taus[1:] - log_taus[:-1]) * fracs[:-1])
            )
        return pd.Series(aup_values)

    def plot(self, figsize=None, model_colors=None, title=None, ax=None):
        """Render Dolan-Moré performance profile curves on a log₁₀(τ) axis.

        Args:
            figsize: Figure size ``(width, height)`` in inches.
            model_colors: Dict mapping model names to colors, or a list in
                model order.
            title: Optional axes title.
            ax: Existing axes to draw into; a new figure is created if
                ``None``.

        Returns:
            matplotlib.figure.Figure: The rendered figure.

        Example:
            >>> result = bench.performance_profiles()
            >>> fig = result.plot(figsize=(8, 5))
        """
        from evaluma.plot import plot_performance_profiles

        return plot_performance_profiles(
            self.table, figsize=figsize, model_colors=model_colors, title=title, ax=ax
        )
