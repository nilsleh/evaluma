from functools import cached_property

import pandas as pd

from evaluma.normalize import normalize


class Benchmark:
    """Container for a normalized model-vs-dataset score matrix.

    After construction the normalized scores are available as ``scores_``.
    Use the analysis methods to compute rankings, comparisons, and profiles.
    """

    def __init__(
        self,
        raw_matrix: pd.DataFrame,
        *,
        norm_ref_low=None,
        norm_ref_high=None,
        metric_direction=None,
        raw_runs=None,
    ):
        """Initialize and normalize the score matrix.

        Args:
            raw_matrix: Model × dataset score matrix (models as row index,
                datasets as columns).
            norm_ref_low: Lower normalization reference — scalar, model
                name, or per-dataset dict. ``None`` triggers data-dependent
                bounds and a ``UserWarning``.
            norm_ref_high: Upper normalization reference, same format as
                ``norm_ref_low``.
            metric_direction: Dict mapping dataset names to ``"min"`` or
                ``"max"``; datasets mapped to ``"min"`` are negated before
                normalization.
            raw_runs: Long-format DataFrame with columns
                ``["model", "dataset", "seed", "score"]`` containing
                per-seed scores. When provided, ``iqm_ranking()`` uses
                stratified bootstrap over seeds.
        """
        self._raw = raw_matrix
        self._norm_ref_low = norm_ref_low
        self._norm_ref_high = norm_ref_high
        self._metric_direction = metric_direction
        self._raw_runs = raw_runs

    def _normalize(self, matrix):
        import warnings

        with warnings.catch_warnings():
            if self._norm_ref_low is not None or self._norm_ref_high is not None:
                warnings.simplefilter("ignore", UserWarning)
            return normalize(
                matrix,
                norm_ref_low=self._norm_ref_low,
                norm_ref_high=self._norm_ref_high,
                metric_direction=self._metric_direction,
            )

    @cached_property
    def scores_(self):
        """Normalized model × dataset score matrix."""
        return self._normalize(self._raw)

    def _new(self, raw_matrix, raw_runs=None):
        return Benchmark(
            raw_matrix,
            norm_ref_low=self._norm_ref_low,
            norm_ref_high=self._norm_ref_high,
            metric_direction=self._metric_direction,
            raw_runs=raw_runs,
        )

    def select_models(self, models):
        """Subset the benchmark to the given models.

        Args:
            models: List of model names to retain.

        Returns:
            Benchmark: New benchmark containing only the selected models.
        """
        raw_runs = None
        if self._raw_runs is not None:
            raw_runs = self._raw_runs[
                self._raw_runs["model"].isin(models)
            ].reset_index(drop=True)
        return self._new(self._raw.loc[models], raw_runs=raw_runs)

    def select_datasets(self, datasets):
        """Subset the benchmark to the given datasets.

        Args:
            datasets: List of dataset names to retain.

        Returns:
            Benchmark: New benchmark containing only the selected datasets.
        """
        raw_runs = None
        if self._raw_runs is not None:
            raw_runs = self._raw_runs[
                self._raw_runs["dataset"].isin(datasets)
            ].reset_index(drop=True)
        return self._new(self._raw[datasets], raw_runs=raw_runs)

    def drop_incomplete(self):
        """Remove models that have missing scores for any dataset.

        Returns:
            Benchmark: New benchmark with incomplete models removed.
        """
        complete = self._raw.index[~self._raw.isna().any(axis=1)]
        raw_runs = None
        if self._raw_runs is not None:
            raw_runs = self._raw_runs[
                self._raw_runs["model"].isin(complete)
            ].reset_index(drop=True)
        return self._new(self._raw.loc[complete], raw_runs=raw_runs)

    def iqm_ranking(self, n_bootstrap=1000, random_state=None):
        """Compute IQM rankings with stratified bootstrap confidence intervals.

        Implements the Agarwal et al. 2021 (rliable) IQM on the flat
        run×dataset score array. Requires multiple seeds; use
        ``aggregate_ranking()`` for single-run data.

        Args:
            n_bootstrap: Number of bootstrap samples for the 95 % CI.
            random_state: Seed for the random number generator.

        Returns:
            IQMResult: Result with ``.table`` and ``.plot()``.

        Raises:
            ValueError: If no seed data is available (``_raw_runs is None``).
        """
        if self._raw_runs is None:
            raise ValueError(
                "iqm_ranking() requires multiple seeds — "
                "use aggregate_ranking() for single-run data."
            )
        from evaluma.methods.iqm import compute_iqm
        from evaluma.normalize import _resolve_bound

        low = _resolve_bound(self._raw, self._norm_ref_low, use_min=True)
        high = _resolve_bound(self._raw, self._norm_ref_high, use_min=False)
        return compute_iqm(
            self._raw_runs,
            norm_bounds=(low, high, self._metric_direction),
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

    def aggregate_ranking(self, agg="trimmed_mean"):
        """Compute a point-estimate descriptive ranking (no CI).

        Works on any benchmark regardless of whether seed data is present.

        Args:
            agg: Aggregation mode — ``"trimmed_mean"`` (default), ``"mean"``,
                or ``"median"``.

        Returns:
            AggregateResult: Result with ``.table`` and ``.plot()``.

        Raises:
            ValueError: If ``agg`` is not a supported mode.
        """
        from evaluma.methods.aggregate import compute_aggregate

        return compute_aggregate(self.scores_, agg=agg)

    def bayesian_comparison(
        self, rope=0.01, reference=None, pairs=None, random_state=None
    ):
        """Compute pairwise Bayesian comparisons via signed-rank test.

        Args:
            rope: Region of practical equivalence half-width.
            reference: If given, only compare each other model against this
                one.
            pairs: Explicit list of ``(model_a, model_b)`` pairs to test.
                Overrides ``reference``.
            random_state: Seed for baycomp's sampler.

        Returns:
            BayesianResult: Result with ``.table`` and ``.plot()``.
        """
        from evaluma.methods.bayesian import compute_bayesian

        return compute_bayesian(
            self.scores_,
            rope=rope,
            reference=reference,
            pairs=pairs,
            random_state=random_state,
        )

    def performance_profiles(self):
        """Compute Dolan-Moré performance profiles.

        Returns:
            ProfileResult: Result with ``.table`` and ``.plot()``.

        Raises:
            ValueError: If any raw score is zero or negative.
        """
        from evaluma.methods.profiles import compute_profiles

        return compute_profiles(self._raw, metric_direction=self._metric_direction)
