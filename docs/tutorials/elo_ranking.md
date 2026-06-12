---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Ranking Models by Head-to-Head Dominance: ELO from TabArena

A model can lead an aggregate ranking while losing most of its direct pairwise comparisons. This happens when a small number of exceptional datasets pull its mean score above competitors that perform more consistently across the rest of the benchmark. Aggregate statistics summarize trimmed performance across all tasks; ELO-style ranking asks a different question: which model wins most often when placed head-to-head against each opponent on each dataset? The method is adapted from [TabArena](https://arxiv.org/abs/2506.16791) (Erickson et al., NeurIPS 2025), where it was introduced as a complement to aggregate ranking on tabular benchmarks.

:::{note}
This tutorial covers ELO ranking from pairwise battle outcomes. For IQM with bootstrap confidence intervals, including the GeoBench data loading pattern reused here, see the [IQM ranking tutorial](iqm_ranking.md). For probabilistic pairwise statements (how likely is Model-A to outperform Model-B on a new dataset?) see the [Bayesian comparison tutorial](bayesian_comparison.md).
:::

```{code-cell} python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluma
```

## 1. Battles: the atomic unit

A **battle** is a single pairwise comparison between two models on one dataset: one model wins (outcome 1), the other loses (outcome 0), or they draw (outcome 0.5). With $M$ models and $N$ datasets there are $\binom{M}{2} \times N$ battles in total. Equal-dataset weighting is enforced by assigning each dataset a total weight of 1 distributed across all model pairs that contested it, so a dataset with many competing models does not dominate the ELO fit.

To illustrate the gap between aggregate ranking and head-to-head dominance, the toy benchmark below places one model at ≈ 0.97 on two datasets while scoring below all others on the remaining five.

```{code-cell} python
rng = np.random.RandomState(42)
datasets = [f"D{i:02d}" for i in range(1, 8)]

scores_dict = {
    "Model-A": np.concatenate([rng.uniform(0.96, 0.98, 2), rng.uniform(0.49, 0.51, 5)]),
    "Model-B": np.concatenate([rng.uniform(0.07, 0.09, 2), rng.uniform(0.71, 0.73, 5)]),
    "Model-C": np.concatenate([rng.uniform(0.05, 0.07, 2), rng.uniform(0.61, 0.63, 5)]),
    "Model-D": np.concatenate([rng.uniform(0.03, 0.05, 2), rng.uniform(0.51, 0.53, 5)]),
}

rows = [
    {"model": m, "dataset": d, "metric": "acc", "score": float(s)}
    for m, sc in scores_dict.items()
    for d, s in zip(datasets, sc)
]
df_toy = pd.DataFrame(rows)

bench = evaluma.load_df(
    df_toy,
    model="model", dataset="dataset", metric="metric", score="score",
    norm_ref_low=0.0, norm_ref_high=1.0,
)
bench.scores_.round(3)
```

:::{margin}
Scores are already in [0, 1] with explicit bounds, so normalization is the identity. The `norm_ref_low=0.0, norm_ref_high=1.0` call is still required to suppress the data-driven-bounds warning.
:::

Model-A scores ≈ 0.97 on D01 and D02 but ≈ 0.50 on D03–D07, where Models B, C, and D all score higher. Model-B is weak on D01–D02 (≈ 0.08) but leads D03–D07 at ≈ 0.72.

Mean aggregation ranks Model-A first because adding two near-perfect scores to five middling ones produces a higher average than adding two very low scores to five strong ones.

```{code-cell} python
mean_result = bench.aggregate_ranking(agg="mean")
mean_result.table.round(3)
```

Model-A's mean score (0.633) leads the field. Model-B, which beats Model-A on five of seven datasets, sits second at 0.539. The aggregate flattens a structure that the win-rate matrix makes visible.

## 2. Win-rate matrix

`bench.elo_ranking()` computes ELO ratings and simultaneously builds the win-rate matrix from all pairwise battles.

```{code-cell} python
result = bench.elo_ranking(n_bootstrap=1000, random_state=42)
```

`result.plot_winrate()` renders the M×M fraction-of-datasets-won for every model pair.

```{code-cell} python
fig = result.plot_winrate(
    figsize=(5, 4),
    title="Win-rate matrix: fraction of datasets where row model beats column model",
)
plt.tight_layout()
plt.show()
```

Model-A's row is 0.29 across every column: it wins only on D01 and D02 (2 out of 7 datasets = 0.29) regardless of which other model it faces. Model-B's row is the opposite: 0.71 against Model-A (winning on D03–D07) and 1.00 against Model-C and Model-D, which it beats on all seven datasets.

The matrix also surfaces a counterintuitive result. Model-D, which ranks last by mean score (0.381), beats Model-A on five of seven datasets (D03–D07, where D ≈ 0.52 and A ≈ 0.50). A model that leads by aggregate can lose to last-place competitors when its high mean is driven by a small number of datasets where it dominates and everyone else collapses.

## 3. MLE ELO ratings with bootstrap CIs

ELO ratings are fit by maximum likelihood, following [TabArena](https://arxiv.org/abs/2506.16791). The ELO model sets the probability that model $i$ beats model $j$ to $1 / \left(1 + \text{base}^{-(R_i - R_j)/\text{scale}}\right)$, where $R$ is the vector of ratings and the constants $\text{base}=10$, $\text{scale}=400$ fix the convention that a 400-point gap corresponds to 10:1 odds. This is a logistic regression: each battle contributes one row with $+\ln(\text{base})$ in model $i$'s column and $-\ln(\text{base})$ in model $j$'s column, weighted to enforce equal dataset contributions. Folding $\ln(\text{base})$ into the design matrix (rather than using $\pm 1$ entries) makes the fitted coefficients land directly on the ELO scale, so the final ratings are simply $\text{scale} \times \text{coefficients}$. Ties are expanded into two half-weight rows, one with outcome 1 and one with outcome 0, so that a draw contributes half a win and half a loss to each side.

Bootstrap confidence intervals resample battles within each dataset. For each of the 1000 replicates, the set of battles that belong to dataset $k$ is resampled with replacement; ELO is then refit on the reassembled battles table. This captures uncertainty about which datasets drive the ranking: if a few datasets dominate a model's ELO, the CI will be wide.

`result.plot()` renders the ratings as a horizontal bar chart with 95% CI error bars.

```{code-cell} python
fig = result.plot(
    figsize=(6, 3.5),
    title="ELO rankings with 95% bootstrap CIs",
)
plt.tight_layout()
plt.show()
```

`result.table` lists the point estimates and 95% CI bounds in descending ELO order.

```{code-cell} python
result.table.round(1)
```

Model-B leads at ELO 1312, well above Model-C (1037) and Model-A (842), which sits close to Model-D (808) despite having the highest mean score. Model-B's CI is wide: bootstrap replicates that draw mainly from D01–D02 (where B scores only ≈ 0.08) produce very low resampled ratings, while replicates that draw mainly from D03–D07 produce very high ones. Model-C and Model-A have overlapping CIs, so the data do not confidently separate their ELO positions.

## 4. `tie_threshold`: treating near-ties as draws

By default, any non-zero score difference produces a decisive battle. A normalized difference of 0.02 is treated the same as one of 0.50, even though the first is likely within measurement noise. `tie_threshold` sets a minimum gap: pairs whose normalized scores differ by at most the threshold produce a draw rather than a win or loss.

`bench.elo_ranking(tie_threshold=0.05)` converts 10 of the 42 battles to draws (the A vs D pairs on D03–D07 and several C–D pairs where scores differ by less than 0.05 on the [0, 1] normalized scale).

```{code-cell} python
result_t = bench.elo_ranking(tie_threshold=0.05, random_state=42)
result_t.table.round(1)
```

The ELO spread narrows: Model-B drops from 1312 to 1224 as some of its dominant wins are softened to partial credit, and the other models follow a similar pattern. The ranking is unchanged.

:::{note}
`tie_threshold` operates on the [0, 1] normalized score scale. A value of 0.05 means "within 5 percentage points on the normalized benchmark scale." This is analogous to the `rope` parameter in `bench.bayesian_comparison()`, which also treats normalized-score differences below a threshold as practically equivalent.
:::

## 5. `calibration_model`: anchoring the scale

Raw ELO ratings carry no inherent absolute meaning: the scale depends on the initialization constant and the specific battles. `calibration_model` shifts all ratings so that a designated anchor lands at exactly 1000, making results interpretable relative to a known baseline and comparable across benchmark runs.

```{code-cell} python
result_c = bench.elo_ranking(calibration_model="Model-B", random_state=42)
result_c.table.round(1)
```

With Model-B anchored to 1000, Model-C sits at 726 and Model-A at 530. The calibration anchor's CI collapses to [1000, 1000] by construction; the CIs for other models widen because they now carry all the fitting uncertainty relative to that fixed point.

## 6. Multi-seed battles with `raw_runs`

When the benchmark is loaded with a seed column, `elo_ranking()` generates one set of battles per (dataset, seed) combination rather than one per dataset. Each dataset still contributes total weight 1 regardless of how many seeds are available, so a dataset with three seeds does not dominate one with one seed. Bootstrap resampling draws within (dataset, seed) groups, which typically narrows CIs when per-seed variance is low.

The multi-seed benchmark below extends the same 4-model, 7-dataset structure with 3 seeds per cell and model-specific per-seed variance.

```{code-cell} python
rng2 = np.random.RandomState(99)
outlier_datasets = {"D01", "D02"}
model_sigma = {
    "Model-A": {"outlier": 0.02, "stable": 0.04},
    "Model-B": {"outlier": 0.01, "stable": 0.02},
    "Model-C": {"outlier": 0.01, "stable": 0.01},
    "Model-D": {"outlier": 0.01, "stable": 0.03},
}

rows_seeded = []
for m, sc in scores_dict.items():
    sigs = model_sigma[m]
    for d, base in zip(datasets, sc):
        sigma = sigs["outlier"] if d in outlier_datasets else sigs["stable"]
        for seed_id in [1, 2, 3]:
            score = float(np.clip(base + rng2.normal(0, sigma), 0.0, 1.0))
            rows_seeded.append(
                {"model": m, "dataset": d, "metric": "acc", "score": score, "seed": seed_id}
            )
df_seeded = pd.DataFrame(rows_seeded)

bench_runs = evaluma.load_df(
    df_seeded,
    model="model", dataset="dataset", metric="metric", score="score",
    seed="seed",
    norm_ref_low=0.0, norm_ref_high=1.0,
)
```

:::{margin}
Passing `seed="seed"` tells evaluma to retain all per-seed rows as `raw_runs`. `bench_runs.scores_` still exposes a mean-aggregated matrix; `elo_ranking()` operates on the full run-level data.
:::

`bench_runs.elo_ranking()` automatically uses the per-seed battles path because `bench_runs` carries raw runs.

```{code-cell} python
result_runs = bench_runs.elo_ranking(random_state=42)

fig = result_runs.plot(
    figsize=(6, 3.5),
    title="ELO with per-seed battles (3 seeds × 7 datasets), 95% CI",
)
plt.tight_layout()
plt.show()
```

`result_runs.table` shows how the CI bounds compare to the single-seed run.

```{code-cell} python
result_runs.table.round(1)
```

Compared to the single-seed run, all CIs are narrower. With 3 seeds per dataset, each dataset contributes 3 groups of battles to the bootstrap rather than 1; resampling within 21 groups (3 seeds × 7 datasets) rather than 7 constrains the estimate more tightly. The ranking is unchanged: Model-B leads and Model-A ranks third despite its high mean score.

## 7. Applying to GeoBenchV2

[GeoBenchV2](https://arxiv.org/abs/2511.15658) (Simumba et al., 2026) evaluates 14 pretrained backbone models on 19 geospatial datasets. The data loading is identical to the IQM tutorial; `elo_ranking()` automatically uses the per-seed battles path because `bench_geo` is loaded with `seed="Seed"`.

```{code-cell} python
df_raw = pd.read_csv("../../results_and_parameters.csv")

full_coverage = (
    df_raw.groupby("backbone")["dataset"]
    .nunique()
    .pipe(lambda s: s[s == 19].index)
    .tolist()
)
df_geo = df_raw[df_raw["backbone"].isin(full_coverage)].copy()

bench_geo = evaluma.load_df(
    df_geo,
    model="backbone",
    dataset="dataset",
    metric="Metric",
    score="test metric",
    seed="Seed",
    norm_ref_low=0.0,
    norm_ref_high=1.0,
    metric_direction={"biomassters": "min"},
)
```

`bench_geo.elo_ranking()` fits MLE ELO from per-seed battles across 19 datasets and 14 backbones. The example uses 200 bootstrap replicates to keep docs execution time bounded; for a final analysis, increase `n_bootstrap` if you need tighter CI estimates.

```{code-cell} python
elo_geo = bench_geo.elo_ranking(n_bootstrap=200, random_state=42)

fig = elo_geo.plot(
    figsize=(9, 5),
    title="GeoBench — ELO ranking with 95% bootstrap CIs (19 datasets)",
)
plt.tight_layout()
plt.show()
```

`elo_geo.plot_winrate()` renders the 14×14 pairwise win-rate heatmap.

```{code-cell} python
fig = elo_geo.plot_winrate(
    figsize=(12, 11),
    title="GeoBench — win-rate matrix (fraction of datasets where row beats column)",
)
plt.tight_layout()
plt.show()
```

### IQM rank vs ELO rank

Placing IQM and ELO rankings side by side identifies backbones whose aggregate score and head-to-head consistency tell different stories.

```{code-cell} python
iqm_geo = bench_geo.iqm_ranking(random_state=42)

elo_ranks = elo_geo.table[["model", "ELO"]].copy()
elo_ranks.insert(0, "elo_rank", range(1, len(elo_ranks) + 1))

iqm_ranks = iqm_geo.table[["model", "IQM"]].copy()
iqm_ranks.insert(0, "iqm_rank", range(1, len(iqm_ranks) + 1))

comparison = elo_ranks.merge(iqm_ranks, on="model")
comparison["rank_diff"] = comparison["elo_rank"] - comparison["iqm_rank"]
comparison["ELO"] = comparison["ELO"].round(1)
comparison["IQM"] = comparison["IQM"].round(3)
comparison.sort_values("elo_rank").reset_index(drop=True)
```

The bottom ten positions are stable across both methods. The top four are reshuffled. By IQM, `convnext_xlarge_fb_in22k` leads narrowly (0.544) over `convnext_large_fb_in22k` (0.543), with `dinov3_convnext_large` third (0.542). By ELO, `dinov3_convnext_large` leads and `convnext_large_fb_in22k` drops to fourth.

The IQM differences at the top are under 0.002, so the ELO reversal reflects genuine head-to-head consistency rather than rounding noise in a tight aggregate race. `convnext_large_fb_in22k` ranks second by IQM but wins fewer head-to-head battles than `dinov3_convnext_large` and `dinov3_vitl16`. The win-rate matrix identifies which specific dataset comparisons drive this split.

When ELO and IQM agree on a backbone's position, that position is robust to both the choice of aggregation method and the distribution of head-to-head outcomes. When they disagree, the win-rate matrix is the right tool to investigate whether the disagreement is driven by a few specialized datasets or a broader pattern across the benchmark.

## Summary

ELO ranking and IQM aggregate capture different aspects of benchmark performance: IQM measures trimmed mean score while ELO measures head-to-head dominance. When they agree the ranking is robust; when they disagree, the win-rate matrix identifies which datasets drive the split.

- Read the win-rate matrix before reading ELO ratings. A model with a high average win-rate is consistently competitive; a uniform low win-rate row signals that the model's aggregate score is driven by a few exceptional datasets rather than broad performance.
- `tie_threshold` prevents trivially small score differences from producing decisive battle outcomes. Set it to a value consistent with your metric's measurement precision, on the [0, 1] normalized scale.
- `calibration_model` fixes the absolute scale so ratings are interpretable relative to a known baseline, useful when comparing across benchmark runs or communicating to a wider audience.
- When `raw_runs` are available (multiple seeds per dataset), `elo_ranking()` automatically generates per-seed battles and resamples within (dataset, seed) groups, which typically narrows CIs compared to single-seed bootstrapping.
- For a probabilistic pairwise statement ("how likely is Model-X to outperform Model-Y on a new task?") use `bench.bayesian_comparison()` as shown in the [Bayesian comparison tutorial](bayesian_comparison.md).

## References

- Erickson, N., Purucker, L., Tschalzev, A., Holzmüller, D., Mutalik Desai, P., Salinas, D., & Hutter, F. (2025). [TabArena: A Living Benchmark for Machine Learning on Tabular Data.](https://arxiv.org/abs/2506.16791) *NeurIPS 2025 Datasets and Benchmarks Track* (Spotlight). [GitHub](https://github.com/autogluon/tabarena)
- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). [Deep reinforcement learning at the edge of the statistical precipice.](https://arxiv.org/abs/2108.13264) *Advances in Neural Information Processing Systems, 34*.
- Simumba, N. et al. (2026). [GEO-Bench: Toward Foundation Models for Earth Monitoring.](https://arxiv.org/abs/2511.15658)
