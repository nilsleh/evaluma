from functools import cached_property

import numpy as np
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
        dataset_metric_map=None,
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
            dataset_metric_map: Dict mapping dataset names to their metric
                name, used by ``improvability_ranking()`` to resolve each
                dataset's error optimum from the metric registry.
        """
        self._raw = raw_matrix
        self._norm_ref_low = norm_ref_low
        self._norm_ref_high = norm_ref_high
        self._metric_direction = metric_direction
        self._raw_runs = raw_runs
        self._dataset_metric_map = dataset_metric_map

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
        """Build a subset Benchmark with normalization bounds frozen from the parent.

        Subsetting filters cells without re-scaling the retained scores: the
        parent's bounds are resolved to concrete per-dataset ``Series`` (on the
        pre-inversion raw matrix) and restricted to the surviving columns, so a
        survivor's normalized score is identical whether or not its peers were
        dropped. ``normalize`` still applies any ``metric_direction`` inversion
        once on these frozen bounds.
        """
        from evaluma.normalize import _resolve_bound

        cols = raw_matrix.columns
        low = _resolve_bound(self._raw, self._norm_ref_low, use_min=True).loc[cols]
        high = _resolve_bound(self._raw, self._norm_ref_high, use_min=False).loc[cols]
        return Benchmark(
            raw_matrix,
            norm_ref_low=low,
            norm_ref_high=high,
            metric_direction=self._metric_direction,
            raw_runs=raw_runs,
            dataset_metric_map=self._dataset_metric_map,
        )

    @property
    def models_(self):
        """Model names in row order."""
        return self._raw.index.tolist()

    @property
    def datasets_(self):
        """Dataset names in column order."""
        return self._raw.columns.tolist()

    def select_models(self, models):
        """Subset the benchmark to the given models.

        Subsetting filters cells without re-scaling retained scores; the
        normalization bounds are frozen from the parent.

        Args:
            models: List of model names to retain.

        Returns:
            Benchmark: New benchmark containing only the selected models.
        """
        raw_runs = None
        if self._raw_runs is not None:
            raw_runs = self._raw_runs[self._raw_runs["model"].isin(models)].reset_index(
                drop=True
            )
        return self._new(self._raw.loc[models], raw_runs=raw_runs)

    def drop_models(self, exclude):
        """Subset the benchmark by dropping specific models.

        Subsetting filters cells without re-scaling retained scores; the
        normalization bounds are frozen from the parent.

        Args:
            exclude: List of model names to remove.

        Returns:
            Benchmark: New benchmark without the excluded models.
        """
        keep = [m for m in self.models_ if m not in set(exclude)]
        return self.select_models(keep)

    def select_datasets(self, datasets):
        """Subset the benchmark to the given datasets.

        Subsetting filters cells without re-scaling retained scores; the
        normalization bounds are frozen from the parent.

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

    def drop_datasets(self, exclude):
        """Subset the benchmark by dropping specific datasets.

        Subsetting filters cells without re-scaling retained scores; the
        normalization bounds are frozen from the parent.

        Args:
            exclude: List of dataset names to remove.

        Returns:
            Benchmark: New benchmark without the excluded datasets.
        """
        keep = [d for d in self.datasets_ if d not in set(exclude)]
        return self.select_datasets(keep)

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

        .. note::
            This is a **descriptive point estimate only** (no CI). The
            trimmed-mean variant trims across datasets, not across seeds; with
            fewer than ~10 datasets the 25% trim is aggressive (e.g. 5
            datasets → only 3 contribute). Treat results as exploratory. For a
            statistically grounded ranking with uncertainty, use
            ``iqm_ranking()`` (requires multiple seeds).

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

    def improvability_ranking(self):
        """Rank models by mean improvability (distance from the per-dataset best).

        For each model, reports the average percent error reduction needed to
        match the best method on each dataset, faithful to the TabArena /
        BeyondArena definition. Error is reconstructed in raw score space from
        each dataset's metric direction and theoretical optimum (from the metric
        registry) — never from the normalized ``scores_`` matrix. Lower is
        better; the per-dataset best method scores ``0``.

        Optima are resolved lazily here (not at load time), so benchmarks whose
        metrics have no defined optimum still load and serve other methods.

        Returns:
            ImprovabilityResult: Result with ``.table``, ``.per_dataset``, and
                ``.plot()``.

        Raises:
            ValueError: If a dataset's metric is not in the registry and is not
                explicitly overridden as ``"min"`` in ``metric_direction``, or
                if its error optimum cannot be resolved.
        """
        from evaluma.methods.improvability import compute_improvability
        from evaluma.metric_registry import get_direction, get_error_optimum

        override = self._metric_direction or {}
        metric_map = self._dataset_metric_map or {}

        direction_map = {}
        optimum_map = {}
        for dataset in self._raw.columns:
            metric_name = metric_map.get(dataset)
            if override.get(dataset) == "min":
                # A "min" column is an error already bottoming out at 0, so its
                # metric name (possibly a raw score name) must not force a 1.0
                # optimum: honor the overridden direction here.
                direction, optimum = "min", 0.0
            elif metric_name is None:
                raise ValueError(
                    f"No metric known for dataset '{dataset}'; cannot compute "
                    "improvability. Improvability currently supports known "
                    "registry metrics, or datasets explicitly marked "
                    "metric_direction='min' so they are treated as error "
                    "columns with optimum 0."
                )
            else:
                try:
                    direction = override.get(dataset) or get_direction(metric_name)
                    optimum = get_error_optimum(metric_name)
                except ValueError as e:
                    raise ValueError(
                        f"Cannot compute improvability for dataset '{dataset}' "
                        f"(metric '{metric_name}'): {e} Improvability currently "
                        "supports known registry metrics, or datasets "
                        "explicitly marked metric_direction='min' so they are "
                        "treated as error columns with optimum 0."
                    ) from e
            direction_map[dataset] = direction
            optimum_map[dataset] = optimum

        return compute_improvability(self._raw, direction_map, optimum_map)

    def bayesian_comparison(
        self, rope=0.01, reference=None, pairs=None, random_state=None
    ):
        """Compute pairwise Bayesian comparisons via signed-rank test.

        Args:
            rope: Region of practical equivalence half-width **in normalized
                score space (0–1)**. Differences smaller than ``rope`` are
                treated as practically equivalent.
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

    def frequentist_comparison(self, reference=None, alpha=0.05):
        """Compute frequentist model comparisons.

        All-pairs mode follows the Demšar (2006) / autorank Friedman + Nemenyi
        workflow. Reference mode is an evaluma extension: pairwise Wilcoxon
        signed-rank tests against a named baseline with Holm correction.

        Runs a Friedman omnibus test first, then either Nemenyi post-hoc
        (all-pairs mode) or Wilcoxon + Holm correction (reference mode).

        Args:
            reference: If given, only compare each other model against this one
                using Wilcoxon + Holm. ``None`` triggers all-pairs Nemenyi mode.
            alpha: Significance level for the ``significant`` column (default 0.05).

        Returns:
            FrequentistResult: Result with ``.table`` and ``.plot()``.

        Raises:
            ValueError: If fewer than 5 datasets are present.

        References:
            Demšar, J. (2006). Statistical Comparisons of Classifiers over
            Multiple Data Sets. *JMLR*, 7, 1–30.
        """
        from evaluma.methods.frequentist import compute_frequentist

        return compute_frequentist(self.scores_, reference=reference, alpha=alpha)

    def elo_ranking(
        self,
        n_bootstrap=1000,
        random_state=None,
        tie_threshold=None,
        calibration_model=None,
    ):
        """Compute MLE ELO rankings with battle-within-task bootstrap CIs.

        Derives a scalar ELO rating per model from pairwise win/loss battles
        across datasets. Each dataset contributes equally via sample weighting.
        Complements ``aggregate_ranking()`` and ``iqm_ranking()`` with a
        pairwise-derived ranking.

        Args:
            n_bootstrap: Number of bootstrap replicates for 95% CI. Set to 0
                to skip bootstrap (CI columns will be NaN).
            random_state: Seed for the random number generator.
            tie_threshold: Minimum score difference (on the [0,1] normalized
                scale) to emit a battle. Pairs within the threshold are skipped.
                ``None`` means strict inequality.
            calibration_model: If given, shift all ratings so this model has
                ELO = 1000.

        Returns:
            EloResult: Result with ``.table``, ``.winrate_matrix``, ``.plot()``,
                and ``.plot_winrate()``.

        Raises:
            ValueError: If ``calibration_model`` is not in the score matrix.
        """
        from evaluma.methods.elo import compute_elo
        from evaluma.normalize import _resolve_bound
        from evaluma.results import EloResult

        # When seed-level battles are used, normalize per-seed scores with the
        # same bounds as ``scores_`` so ``tie_threshold`` is on the [0,1] scale,
        # consistent with the win-rate matrix (built from ``scores_``).
        norm_bounds = None
        if self._raw_runs is not None:
            low = _resolve_bound(self._raw, self._norm_ref_low, use_min=True)
            high = _resolve_bound(self._raw, self._norm_ref_high, use_min=False)
            norm_bounds = (low, high)

        table, winrate_matrix = compute_elo(
            self.scores_,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            tie_threshold=tie_threshold,
            calibration_model=calibration_model,
            raw_runs=self._raw_runs,
            metric_direction=self._metric_direction,
            norm_bounds=norm_bounds,
        )
        return EloResult(table, winrate_matrix)

    def performance_profiles(self):
        """Compute Dolan-Moré performance profiles.

        Profiles are computed on the raw (un-normalized) score matrix. All raw
        values must be strictly positive; see ``Raises``.

        Returns:
            ProfileResult: Result with ``.table`` and ``.plot()``.

        Raises:
            ValueError: If any raw score is zero or negative.
        """
        from evaluma.methods.profiles import compute_profiles

        return compute_profiles(self._raw, metric_direction=self._metric_direction)

    def _validate_callable_rank_vector(self, ranks: pd.Series) -> pd.Series:
        """Validate and align a custom ranker's output.

        The callable-ranker contract is intentionally narrow: it must return a
        numeric ``pd.Series`` indexed by this benchmark's models, where lower
        values are better and ``1`` denotes the best rank.

        Args:
            ranks: Candidate rank vector returned by a custom ranker.

        Returns:
            pd.Series: Float rank vector aligned to ``self.models_`` order.

        Raises:
            TypeError: If ``ranks`` is not a numeric ``pd.Series``.
            ValueError: If the series index does not match the benchmark's
                model set, or if any rank value is missing / non-finite.
        """
        if not isinstance(ranks, pd.Series):
            raise TypeError(
                "Custom ranker must return a pandas Series indexed by model."
            )

        expected = set(self.models_)
        actual = set(ranks.index)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing models: {missing}")
            if extra:
                parts.append(f"unexpected models: {extra}")
            raise ValueError(
                "Custom ranker must return ranks for exactly this benchmark's "
                "models; " + "; ".join(parts)
            )

        aligned = ranks.loc[self.models_]
        if not pd.api.types.is_numeric_dtype(aligned):
            raise TypeError("Custom ranker must return numeric rank values.")

        aligned = aligned.astype(float)
        if aligned.isna().any() or not np.isfinite(aligned.to_numpy()).all():
            raise ValueError(
                "Custom ranker must return finite rank values for every model."
            )
        return aligned

    def _rank_vector(self, ranker):
        """Per-model rank Series (rank 1 = best) under a named or custom ranker.

        Args:
            ranker: ``"avg_rank"`` (mean of per-dataset ranks), ``"elo"``
                (MLE ELO rating), ``"improvability"`` (mean error reduction to
                the per-dataset best), or a callable ``bench -> pd.Series``
                returning literal per-model ranks indexed by model.

        Returns:
            pd.Series: Per-model ranks (rank 1 = best) indexed by model.

        Raises:
            TypeError: If a callable ``ranker`` does not return a numeric
                ``pd.Series`` indexed by model.
            ValueError: If ``ranker`` is an unknown name.
        """
        if callable(ranker):
            return self._validate_callable_rank_vector(ranker(self))
        if ranker == "avg_rank":
            keys = self.scores_.rank(ascending=False, axis=0).mean(axis=1)
            return keys.rank(ascending=True, method="average")
        if ranker == "elo":
            keys = self.elo_ranking(n_bootstrap=0).table.set_index("model")["ELO"]
            return keys.rank(ascending=False, method="average")
        if ranker == "improvability":
            table = self.improvability_ranking().table.set_index("model")
            return table["improvability"].rank(ascending=True, method="average")
        raise ValueError(
            f"Unknown ranker {ranker!r}; expected 'avg_rank', 'elo', "
            "'improvability', or a callable."
        )

    def rank_sensitivity(
        self,
        other,
        cond_a,
        cond_b,
        n_bootstrap=1000,
        random_state=None,
        agg="trimmed_mean",
        ranker="aggregate",
    ):
        """Quantify whether rankings reorder between two conditions.

        Args:
            other: Benchmark for condition B.
            cond_a: Label for this benchmark's condition.
            cond_b: Label for ``other`` benchmark's condition.
            n_bootstrap: Number of dataset-bootstrap replicates for 95% CI.
                Only used when ``ranker="aggregate"``.
            random_state: Seed for bootstrap sampling.
            agg: Per-model aggregation defining the ranking when
                ``ranker="aggregate"``. Defaults to ``"trimmed_mean"`` to match
                :meth:`aggregate_ranking`; ``"mean"`` is available for
                light-tailed or very-small-N data, and ``"median"`` is also
                accepted.
            ranker: Ranking method whose two orderings tau compares.
                ``"aggregate"`` (default) uses ``agg`` on the normalized score
                matrix with a bootstrap CI — the original behavior. ``"avg_rank"``,
                ``"elo"``, ``"improvability"``, or a callable
                ``bench -> pd.Series`` decouple the ranking from the aggregate
                family and return a point estimate (``tau_ci=(nan, nan)``).
                Custom callables must return literal numeric ranks indexed by
                model, with lower values better and ``1`` meaning best.

        Returns:
            RankSensitivityResult: Rank sensitivity result object.

        Raises:
            TypeError: If ``other`` is not a :class:`Benchmark`.
            ValueError: If model or dataset sets differ across benchmarks.
        """
        import warnings

        from evaluma.methods.rank_sensitivity import (
            compute_rank_sensitivity,
            compute_rank_sensitivity_from_ranks,
        )

        if not isinstance(other, Benchmark):
            raise TypeError(f"other must be a Benchmark, got {type(other).__name__}.")

        models_a = set(self.scores_.index)
        models_b = set(other.scores_.index)
        missing_in_b = sorted(models_a - models_b)
        missing_in_a = sorted(models_b - models_a)
        if missing_in_a or missing_in_b:
            parts = []
            if missing_in_b:
                parts.append(f"missing from {cond_b}: {missing_in_b}")
            if missing_in_a:
                parts.append(f"missing from {cond_a}: {missing_in_a}")
            raise ValueError("Model mismatch between conditions: " + "; ".join(parts))

        datasets_a = set(self.scores_.columns)
        datasets_b = set(other.scores_.columns)
        missing_ds_in_b = sorted(datasets_a - datasets_b)
        missing_ds_in_a = sorted(datasets_b - datasets_a)
        if missing_ds_in_a or missing_ds_in_b:
            parts = []
            if missing_ds_in_b:
                parts.append(f"missing from {cond_b}: {missing_ds_in_b}")
            if missing_ds_in_a:
                parts.append(f"missing from {cond_a}: {missing_ds_in_a}")
            raise ValueError("Dataset mismatch between conditions: " + "; ".join(parts))

        if ranker != "aggregate":
            label = ranker if isinstance(ranker, str) else "custom"
            if n_bootstrap > 0:
                warnings.warn(
                    f"n_bootstrap={n_bootstrap} is ignored for ranker={label!r}; "
                    "a bootstrap CI is only produced for ranker='aggregate'. "
                    "Returning a point estimate with tau_ci=(nan, nan).",
                    UserWarning,
                    stacklevel=2,
                )
            rank_a = self._rank_vector(ranker)
            rank_b = other._rank_vector(ranker)
            return compute_rank_sensitivity_from_ranks(
                rank_a, rank_b, cond_a, cond_b, agg=label
            )

        if len(datasets_a) < 5:
            msg = f"Only {len(datasets_a)} datasets provided; bootstrap CI may be wide."
            if agg == "trimmed_mean":
                msg += (
                    " With agg='trimmed_mean' the 25% per-dataset trim is degenerate "
                    "at this N (few datasets contribute to each model's score); "
                    "consider agg='mean'."
                )
            warnings.warn(msg, UserWarning, stacklevel=2)

        aligned_other = other.scores_.loc[self.scores_.index, self.scores_.columns]
        return compute_rank_sensitivity(
            self.scores_,
            aligned_other,
            cond_a=cond_a,
            cond_b=cond_b,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            agg=agg,
        )


class BenchmarkGroup:
    """Collection of condition-keyed Benchmark objects."""

    def __init__(self, benchmarks):
        """Initialize a condition-keyed benchmark container.

        Args:
            benchmarks: Mapping from condition label to :class:`Benchmark`.

        Raises:
            ValueError: If fewer than two conditions are provided.
        """
        if len(benchmarks) < 2:
            raise ValueError(
                f"BenchmarkGroup requires at least 2 conditions; got {len(benchmarks)}."
            )
        self._benchmarks = dict(benchmarks)

    def __getitem__(self, key):
        """Return the benchmark for one condition label.

        Args:
            key: Condition label key.

        Returns:
            Benchmark: Benchmark associated with ``key``.

        Raises:
            KeyError: If ``key`` is not present.
        """
        return self._benchmarks[key]

    def rank_sensitivity(
        self,
        cond_a,
        cond_b,
        n_bootstrap=1000,
        random_state=None,
        agg="trimmed_mean",
        ranker="aggregate",
    ):
        """Run rank-sensitivity analysis between two conditions in the group.

        Args:
            cond_a: Condition A label.
            cond_b: Condition B label.
            n_bootstrap: Number of dataset-bootstrap replicates for 95% CI.
                Only used when ``ranker="aggregate"``.
            random_state: Seed for bootstrap sampling.
            agg: Per-model aggregation defining the ranking when
                ``ranker="aggregate"``. Defaults to ``"trimmed_mean"`` to
                match :meth:`Benchmark.aggregate_ranking`; ``"mean"`` is
                available for light-tailed or very-small-N data.
            ranker: Same ranking selector supported by
                :meth:`Benchmark.rank_sensitivity`. ``"aggregate"`` preserves
                the original grouped behavior; alternate rankers return point
                estimates with ``tau_ci=(nan, nan)``.

        Returns:
            RankSensitivityResult: Rank sensitivity result object.

        Raises:
            KeyError: If either condition label is missing.
            ValueError: If the two benchmarks have mismatched model/dataset sets.
        """
        return self[cond_a].rank_sensitivity(
            self[cond_b],
            cond_a=cond_a,
            cond_b=cond_b,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            agg=agg,
            ranker=ranker,
        )

    def select_models(self, models):
        """Select the same model subset across all conditions.

        Args:
            models: List of model names to retain.

        Returns:
            BenchmarkGroup: New group with selected models in each benchmark.
        """
        return BenchmarkGroup(
            {k: b.select_models(models) for k, b in self._benchmarks.items()}
        )

    def drop_models(self, exclude):
        """Drop the same model subset across all conditions.

        Args:
            exclude: List of model names to remove.

        Returns:
            BenchmarkGroup: New group with excluded models removed.
        """
        return BenchmarkGroup(
            {k: b.drop_models(exclude) for k, b in self._benchmarks.items()}
        )

    def select_datasets(self, datasets):
        """Select the same dataset subset across all conditions.

        Args:
            datasets: List of dataset names to retain.

        Returns:
            BenchmarkGroup: New group with selected datasets in each benchmark.
        """
        return BenchmarkGroup(
            {k: b.select_datasets(datasets) for k, b in self._benchmarks.items()}
        )

    def drop_datasets(self, exclude):
        """Drop the same dataset subset across all conditions.

        Args:
            exclude: List of dataset names to remove.

        Returns:
            BenchmarkGroup: New group with excluded datasets removed.
        """
        return BenchmarkGroup(
            {k: b.drop_datasets(exclude) for k, b in self._benchmarks.items()}
        )
