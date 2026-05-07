# Overview

Given a collection of models evaluated on multiple datasets, the question "which model is best?" is harder than it sounds. Simple averages obscure robustness, ignore statistical uncertainty, and make it difficult to compare methods across papers. However, several works in the literature and different competition benchmarks have explored various schemas to come up with informative rankings. **evaluma** is designed to be an accessible entry point to build such rankings for your benchmark evaluations providing four complementary analyses from a single input format:

- **Aggregate ranking** — point-estimate ranking via trimmed mean, mean, or median
- **IQM ranking** — robust central tendency with bootstrap confidence intervals (requires multiple seeds per model/dataset)
- **Bayesian comparison** — probability that one model outperforms another
- **Performance profiles** — cumulative distribution showing how often each model comes within a factor of the best

This page walks through installing the package, preparing your results CSV, and running the analyses.

## Installation

```bash
pip install evaluma
```

For a development install from source:

```bash
git clone https://github.com/nilsleh/evaluma
cd evaluma
pip install -e ".[dev]"
```

## Input format

evaluma expects a **long-format CSV** with one row per (model, dataset) combination:

| model | dataset | metric | score |
|-------|---------|--------|-------|
| ResNet | CIFAR10 | acc | 0.91 |
| ResNet | ImageNet | acc | 0.87 |
| ViT | CIFAR10 | acc | 0.94 |
| ViT | ImageNet | acc | 0.89 |
| ConvNeXt | CIFAR10 | acc | 0.93 |
| ConvNeXt | ImageNet | acc | 0.85 |

The four required columns are:

| Column | Description |
|--------|-------------|
| `model` | Model or method name |
| `dataset` | Dataset or task name |
| `metric` | Metric name (e.g., `acc`, `iou`, `rmse`) |
| `score` | The numeric score value |

Multiple random seeds per (model, dataset) are supported — add a `seed` column and pass its name to `load_csv()`.

If your CSV uses different column names, pass them as arguments to `load_csv()` — see the quickstart below.

## Quickstart

Load your CSV and construct a `Benchmark` object. The column arguments tell evaluma which CSV columns map to model, dataset, metric, and score:

```python
import evaluma

bench = evaluma.load_csv(
    "results.csv",
    model="model",
    dataset="dataset",
    metric="metric",
    score="score",
)
```

The `Benchmark` object holds your data and provides four analysis methods. Each returns a result object with a `.table` (pandas DataFrame) and a `.plot()` method:

### Aggregate ranking

Compute a point-estimate ranking across datasets:

```python
agg = bench.aggregate_ranking()
print(agg.table)
agg.plot()
```

The default aggregation is the trimmed mean (discards the top and bottom 25% of per-dataset scores). Pass `agg="mean"` or `agg="median"` for alternatives. This method works with any benchmark, including single-run data.

### IQM ranking

Compute the interquartile mean with 95% bootstrap confidence intervals:

```python
iqm = bench.iqm_ranking()
print(iqm.table)
iqm.plot()
```

The IQM discards the top and bottom 25% of per-dataset scores before averaging, making it resistant to outliers. The bootstrap CIs are stratified — seeds are resampled independently within each dataset. **Requires multiple seeds** — pass `seed="seed_column"` to `load_csv()` when loading. Use `aggregate_ranking()` for single-run data. See the [IQM Tutorial](./tutorials/iqm_ranking.md) for a more in-depth example.

### Bayesian pairwise comparison

Compute posterior probabilities for every model pair:

```python
bayes = bench.bayesian_comparison()
print(bayes.table)
bayes.plot()
```

For each pair (A, B), the output gives `p_a_better`, `p_equiv`, and `p_b_better` — the probabilities that A is better, equivalent (within a practical equivalence region), or worse than B. See the [Bayesian Comparison](./tutorials/bayesian_comparison.md) for a more in-depth example.

### Performance profiles

Plot how often each model achieves near-best performance across datasets:

```python
profiles = bench.performance_profiles()
print(profiles.table)
profiles.plot()
```

The profile curve shows, for each performance ratio τ ≥ 1, the fraction of datasets where a model's score is within τ of the best. A curve that rises faster means the model is closer to best more often. See the [Performance Profile Tutorial](./tutorials/performance_profiles.md) for a more in-depth example.

## Column mapping

If your CSV uses different column names, pass them explicitly:

```python
bench = evaluma.load_csv(
    "results.csv",
    model="experiment",
    dataset="task",
    metric="measure",
    score="value",
)
```

## CLI quickstart

Run all three point-estimate analyses and write CSV and PNG outputs to the current directory:

```bash
evaluma report results.csv
```

With custom column names:

```bash
evaluma report results.csv --model experiment --dataset task --score value
```

Save outputs to a specific directory:

```bash
evaluma report results.csv --output ./results/
```

Individual subcommands are also available: `evaluma rank` (IQM with seeds), `evaluma aggregate`, `evaluma compare`, and `evaluma profiles`. The CLI supports the same column mapping with `--model`, `--dataset`, `--metric`, and `--score` flags, or via a YAML config file passed with `--config`.

## Next steps

- [IQM Ranking tutorial](tutorials/iqm_ranking) — deeper dive with worked examples
- [Bayesian Comparison tutorial](tutorials/bayesian_comparison) — interpreting posterior probabilities
- [Performance Profiles tutorial](tutorials/performance_profiles) — understanding the Dolan-Moré framework
