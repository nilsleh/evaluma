# Configuration Guide

The [overview](overview) covers the simplest case: one metric, one run per model, all datasets complete. Real benchmarks often involve multiple evaluation runs per model, a mix of metric types such as accuracy and RMSE, or models that were not evaluated on every dataset. This page covers each of those configuration decisions in turn.

---

## Multiple evaluation runs

When each model is evaluated more than once per dataset — under different random seeds or training runs — you should pass the column that identifies the seed to `load_csv()`:

```python
import evaluma

bench = evaluma.load_csv(
    "results.csv",
    seed="seed",
)
```

evaluma preserves all per-seed rows internally and averages them per (model, dataset) to build the score matrix exposed in `bench.scores_`. The raw per-seed data is stored separately and passed to the analysis methods that use it.

### Why seeds matter for IQM

`iqm_ranking()` requires a seed column. Without one, calling it raises a `ValueError`. The method implements the Agarwal et al. (2021) IQM on the full run × dataset array: all per-seed scores for each model are concatenated into a flat vector, the outer 25% are trimmed, and the remainder is averaged. A stratified bootstrap then constructs 95% confidence intervals by resampling seeds independently within each dataset, capturing how sensitive the ranking is to which particular seeds happened to run.

If your benchmark has only a single run per model per dataset, use `aggregate_ranking()` instead:

```python
result = bench.aggregate_ranking(agg="trimmed_mean")
result.plot()
```

This returns a point estimate with no confidence interval. The [IQM ranking tutorial](tutorials/iqm_ranking) covers both methods in depth.

---

## Mixed metric types

evaluma normalizes all scores to [0, 1] before computing IQM rankings and Bayesian comparisons. The formula is:

```
normalized = (score - ref_low) / (ref_high - ref_low)
```

For lower-is-better metrics the direction is flipped so that 1 always means best:

```
normalized = (ref_high - score) / (ref_high - ref_low)
```

For metrics bounded to [0, 1] such as accuracy, F1, and IoU, evaluma uses `ref_low = 0.0` and `ref_high = 1.0` automatically. For regression metrics like RMSE there is no natural upper bound, so a normalized score cannot be computed without additional information. Passing an RMSE value through as-is against `ref_high = 1.0` produces values greater than 1, which silently corrupts IQM and Bayesian results. The `metric_type_bounds` parameter fixes this.

### The metric registry

evaluma knows these metrics out of the box:

| Metric name(s)               | Direction | Natural bounds |
|------------------------------|-----------|----------------|
| accuracy, acc                | max       | [0.0, 1.0]     |
| iou, miou                    | max       | [0.0, 1.0]     |
| f1                           | max       | [0.0, 1.0]     |
| auc, ap, map                 | max       | [0.0, 1.0]     |
| precision, recall            | max       | [0.0, 1.0]     |
| r2                           | max       | [0.0, 1.0]     |
| rmse, mae, mse, mape         | min       | [0.0, **?**]   |

Metrics marked **?** have no natural upper bound and must appear in `metric_type_bounds`. Metrics with natural [0, 1] bounds are used automatically when not listed.

### Configuring bounds

Pass `metric_type_bounds` as a dict mapping metric names to `(low, high)` tuples. Bounded metrics that are not listed (accuracy, IoU, F1, etc.) fall back to their registry defaults automatically.

**Python API:**

```python
bench = evaluma.load_csv(
    "results.csv",
    metric_type_bounds={
        "rmse": (0.0, 50.0),   # explicit domain ceiling
        "mae":  (0.0, 12.5),
    },
)
```

**YAML config (CLI):**

```yaml
# config.yaml
metric_type_bounds:
  rmse: [0.0, 50.0]
  mae:  [0.0, 12.5]
```

```bash
evaluma report results.csv --config config.yaml --output ./out/
```

`metric_type_bounds` is mutually exclusive with `norm_ref_low` / `norm_ref_high`. Passing both raises `ValueError`.

### Using a baseline model as the ceiling

When the appropriate upper bound for a regression metric depends on your data distribution, you can use the name of a reference model as the ceiling. evaluma looks up that model's score on each dataset independently and uses it as `ref_high` for that dataset:

```python
bench = evaluma.load_csv(
    "results.csv",
    metric_type_bounds={
        "rmse": (0.0, "baseline"),   # baseline model's per-dataset RMSE is the ceiling
    },
)
```

The reference model must be present in the CSV. A model that performs worse than the reference gets a normalized score below 0, which is expected: it means the model is worse than the reference you chose, and the negative value is informative rather than erroneous.

### Performance profiles and metric bounds

Performance profiles (Dolan-Moré) operate on raw scores, not on normalized values. They compute ratios of raw scores across models, so they are scale-invariant and do not require `metric_type_bounds` or normalization bounds of any kind.

The only thing that matters for profiles is optimization direction. When `metric_type_bounds` is provided, direction is inferred from the registry automatically. For custom metrics, add `metric_direction` as well (see below).

### Adding a custom metric

If your metric is not in the registry (for example CRPS or NLL), you must specify both its direction and bounds explicitly:

```python
bench = evaluma.load_csv(
    "results.csv",
    metric_type_bounds={
        "crps": (0.0, 2.0),
    },
    metric_direction={
        "dataset_crps_1": "min",
        "dataset_crps_2": "min",
    },
)
```

`metric_direction` entries take precedence over registry-inferred directions, so you can also use them to correct a registry default without forking the code.

---

## Incomplete score matrices

By default, `load_csv()` raises a `ValueError` if any (model, dataset) cell is missing from the CSV. There are two ways to handle an incomplete matrix.

**Drop at load time** — pass `drop_incomplete=True` to silently remove any model that is missing scores on at least one dataset:

```python
bench = evaluma.load_csv("results.csv", drop_incomplete=True)
```

**Drop after inspection** — load without the flag to get a `Benchmark` containing all models (evaluma raises if the matrix is incomplete, so your CSV must be pre-filtered or use `drop_incomplete=True`), then call `bench.drop_incomplete()` on the result:

```python
# Load only models present in the CSV for every dataset
bench_full = evaluma.load_csv("results.csv", drop_incomplete=True)

# Inspect which models were retained
print(bench_full.scores_.index.tolist())

# Or subset manually before dropping
bench_sub = bench_full.select_models(["ModelA", "ModelB", "ModelC"])
```

`bench.drop_incomplete()` returns a new `Benchmark` with incomplete models removed, leaving the normalization bounds and seed data intact. Use it when you want to examine the full score matrix first and make an explicit decision about which models to include, rather than silently discarding them at load time.
