<p align="center">
<img src="docs/_static/logo.png" alt="evaluma_logo" width="300" height="auto" />
</p>

<p align="center">
<a href="https://github.com/nilsleh/evaluma/actions/workflows/ci.yml"><img src="https://github.com/nilsleh/evaluma/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
<a href="https://codecov.io/gh/nilsleh/evaluma"><img src="https://codecov.io/gh/nilsleh/evaluma/branch/main/graph/badge.svg" alt="Coverage"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
<a href="https://pypi.org/project/evaluma/"><img src="https://img.shields.io/pypi/v/evaluma.svg" alt="PyPI"></a>
<a href="https://nilsleh.github.io/evaluma"><img src="https://img.shields.io/badge/docs-nilsleh.github.io-blue.svg" alt="Docs"></a>
</p>

# evaluma

A small Python package for comparing machine learning models across benchmark suites. Given a CSV of per-model, per-dataset scores, evaluma can compute three complementary views of the results:

- **IQM ranking** — interquartile mean with bootstrapped confidence intervals, following [Agarwal et al. (2021)](https://arxiv.org/abs/2108.13264)
- **Bayesian pairwise comparison** — posterior probabilities that model A beats model B (or is practically equivalent), via [baycomp](https://github.com/janezd/baycomp)
- **Dolan-Moré performance profiles** — cumulative distribution of performance ratios and area-under-profile scores


## Documentation

Full documentation, including tutorials and API reference, is available at
[nilsleh.github.io/evaluma](https://nilsleh.github.io/evaluma).

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

## Quick start

### Python API

```python
import evaluma

bench = evaluma.load_df(
    "results.csv",
    model="model",
    dataset="dataset",
    metric="metric",
    score="score",
)

# IQM ranking with 95% bootstrap CI
iqm = bench.iqm_ranking()
print(iqm.table)
fig = iqm.plot()
fig.savefig("iqm.png")

# Bayesian pairwise probabilities
bayes = bench.bayesian_comparison()
print(bayes.table)

# Dolan-Moré performance profiles
profiles = bench.performance_profiles()
fig = profiles.plot()
```

### CLI

```bash
# Run all three analyses and write six output files
evaluma report results.csv \
    --model model --dataset dataset --metric metric --score score \
    --output results/

# Individual subcommands
evaluma rank    results.csv --model model --dataset dataset --metric metric --score score --output results/
evaluma compare results.csv --model model --dataset dataset --metric metric --score score --output results/
evaluma profiles results.csv --model model --dataset dataset --metric metric --score score --output results/
```

Each subcommand writes a `.csv` table and a `.png` figure to `--output`.

### Column mapping

If your CSV uses different column names, pass them explicitly:

```bash
evaluma report results.csv \
    --model experiment --dataset task --metric measure --score value \
    --output results/
```

Or put them in a YAML config file:

```yaml
# config.yaml
model: experiment
dataset: task
metric: measure
score: value
```

```bash
evaluma report results.csv --config config.yaml --output results/
```

### Lower-is-better metrics

```python
bench = evaluma.load_df(
    "results.csv",
    model="model", dataset="dataset", metric="metric", score="score",
    metric_direction={"rmse": "min"},
)
```

```bash
evaluma report results.csv ... --metric-direction rmse:min
```

### Filtering models or datasets

```python
bench_ab = bench.select_models(["ModelA", "ModelB"])
bench_core = bench.select_datasets(["dataset1", "dataset2", "dataset3"])
```

## Input format

evaluma expects a **long-format CSV** with one row per (model, dataset) combination:

```
model,dataset,metric,score
ModelA,dataset1,acc,0.91
ModelA,dataset2,acc,0.87
ModelB,dataset1,acc,0.84
...
```

Multiple seeds are supported — pass `--seed seed_col` and evaluma aggregates by mean before analysis.

## Contributing

```bash
git clone https://github.com/nilsleh/evaluma
cd evaluma
pip install -e ".[dev]"

# Run tests
pytest --cov=evaluma --cov-report=term-missing

# Lint and format
ruff check .
ruff format .

# Type checking
ty check
```

Bug reports and pull requests are welcome on [GitHub](https://github.com/nilsleh/evaluma).

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text.

## Citation

If you use evaluma in your research, please cite:

```bibtex
@software{lehmann2026evaluma,
  author  = {Lehmann, Nils},
  title   = {evaluma: Holistic ML Benchmark Ranking},
  year    = {2026},
  url     = {https://github.com/nilsleh/evaluma},
  version = {0.1.0},
}
```

also cite the works of the underlying methods and frameworks used:

```bibtex
@inproceedings{agarwal2021deep,
  title     = {Deep Reinforcement Learning at the Edge of the Statistical Precipice},
  author    = {Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel
               and Courville, Aaron and Bellemare, Marc G.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
}

@article{benavoli2017time,
  title   = {Time for a Change: a Tutorial for Comparing Multiple Classifiers
             Through Bayesian Analysis},
  author  = {Benavoli, Alessio and Corani, Giorgio and Dem{\v{s}}ar, Janez
             and Zaffalon, Marco},
  journal = {Journal of Machine Learning Research},
  volume  = {18},
  number  = {77},
  pages   = {1--36},
  year    = {2017},
}

@article{dolan2002benchmarking,
  title   = {Benchmarking Optimization Software with Performance Profiles},
  author  = {Dolan, Elizabeth D. and Mor{\'e}, Jorge J.},
  journal = {Mathematical Programming},
  volume  = {91},
  pages   = {201--213},
  year    = {2002},
}
```
