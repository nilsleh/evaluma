# evaluma

**Holistic ML benchmark ranking** — Interquartile mean, Bayesian pairwise comparison, and Dolan-Moré performance profiles in a single Python API and CLI.

::::{grid} 1 1 3 3
:class-container: text-center

:::{grid-item-card} IQM Ranking
Bootstrap confidence intervals around the interquartile mean, following best practices from deep RL research.
:::

:::{grid-item-card} Bayesian Comparison
Posterior probabilities that one model beats another (or is practically equivalent) across multiple datasets.
:::

:::{grid-item-card} Performance Profiles
Dolan-Moré cumulative profiles showing what fraction of benchmarks a model solves within a given performance ratio.
:::
::::

---

## Quick links

- [Overview & Quickstart](overview) — Install and run your first analysis
- [Configuration Guide](configuration) — Seeds, metric bounds, and incomplete data
- [Tutorials](tutorials/index) — Deep dives into each method
- [API Reference](autoapi/index) — Full module documentation
- [Contributing](contributing) — Set up a dev environment

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

overview
configuration
autoapi/index
tutorials/index
references
contributing
```
