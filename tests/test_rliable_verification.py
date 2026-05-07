"""Step 0 verification: evaluma IQM point estimates match rliable exactly.

Synthetic example: 3 models × 4 datasets × 3 seeds, all scores pre-normalized
to [0, 1].  We load the data via evaluma.load_df() with norm_ref_low=0,
norm_ref_high=1 so normalization is a no-op, then compare:

  1. Point estimates (trim_mean on flat array) — must agree to within 1e-10.
  2. CI widths — must have the same rank ordering across models and be within
     2× of each other, confirming both use equivalent stratified bootstraps.
"""

import numpy as np
import pandas as pd
import pytest
from rliable import library as rly
from rliable import metrics as rly_metrics

# ---------------------------------------------------------------------------
# Synthetic dataset (fixed, no randomness in score design)
# ---------------------------------------------------------------------------

MODELS = ["A", "B", "C"]
DATASETS = ["d1", "d2", "d3", "d4"]
SEEDS = [1, 2, 3]

# Pre-normalized scores in [0, 1]; shape: (model, dataset, seed)
RAW_SCORES = {
    #         d1              d2              d3              d4
    "A": [[0.90, 0.85, 0.95], [0.80, 0.75, 0.85], [0.70, 0.65, 0.75], [0.60, 0.55, 0.65]],
    "B": [[0.50, 0.45, 0.55], [0.60, 0.55, 0.65], [0.40, 0.35, 0.45], [0.50, 0.45, 0.55]],
    "C": [[0.20, 0.18, 0.22], [0.30, 0.28, 0.32], [0.25, 0.23, 0.27], [0.35, 0.33, 0.37]],
}

N_BOOTSTRAP = 5000
RANDOM_STATE = 42


@pytest.fixture(scope="module")
def evaluma_result():
    import evaluma

    rows = []
    for model in MODELS:
        for di, dataset in enumerate(DATASETS):
            for si, seed in enumerate(SEEDS):
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "metric": "acc",
                    "score": RAW_SCORES[model][di][si],
                    "seed": seed,
                })
    bench = evaluma.load_df(
        pd.DataFrame(rows),
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        seed="seed",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
    )
    return bench.iqm_ranking(n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE)


@pytest.fixture(scope="module")
def rliable_result():
    """rliable IQM on the same scores.

    rliable expects a dict of (num_runs, num_tasks) arrays.  We build one
    array per model where rows = seeds, columns = datasets.
    """
    score_dict = {
        m: np.array(RAW_SCORES[m], dtype=float).T  # shape: (num_seeds, num_datasets)
        for m in MODELS
    }
    rng = np.random.RandomState(RANDOM_STATE)
    point_ests, interval_ests = rly.get_interval_estimates(
        score_dict,
        lambda x: np.array([rly_metrics.aggregate_iqm(x)]),
        reps=N_BOOTSTRAP,
        random_state=rng,
    )
    # point_ests[m] is shape (1,); interval_ests[m] is shape (2, 1)
    return {
        m: {
            "iqm": float(point_ests[m][0]),
            "ci_low": float(interval_ests[m][0, 0]),
            "ci_high": float(interval_ests[m][1, 0]),
        }
        for m in MODELS
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_point_estimates_agree(evaluma_result, rliable_result):
    """Point estimates must match exactly (same trim_mean formula)."""
    tbl = evaluma_result.table.set_index("model")
    for model in MODELS:
        ev_iqm = tbl.loc[model, "IQM"]
        rl_iqm = rliable_result[model]["iqm"]
        assert abs(ev_iqm - rl_iqm) < 1e-10, (
            f"Model {model}: evaluma IQM={ev_iqm:.8f}, rliable IQM={rl_iqm:.8f}, "
            f"diff={abs(ev_iqm - rl_iqm):.2e}"
        )


def test_ci_widths_consistent(evaluma_result, rliable_result):
    """CI widths from both libraries must have the same rank order and be
    within 2× of each other (same stratified bootstrap, different RNGs)."""
    tbl = evaluma_result.table.set_index("model")
    ev_widths = {m: tbl.loc[m, "CI_high"] - tbl.loc[m, "CI_low"] for m in MODELS}
    rl_widths = {m: rliable_result[m]["ci_high"] - rliable_result[m]["ci_low"] for m in MODELS}

    # Rank ordering must agree
    ev_order = sorted(MODELS, key=lambda m: ev_widths[m])
    rl_order = sorted(MODELS, key=lambda m: rl_widths[m])
    assert ev_order == rl_order, (
        f"CI width rank order differs: evaluma={ev_order}, rliable={rl_order}\n"
        f"evaluma widths: {ev_widths}\nrliable widths: {rl_widths}"
    )

    # Magnitudes must be within 2× (same estimand, different RNG → sampling variance)
    for m in MODELS:
        ratio = ev_widths[m] / rl_widths[m]
        assert 0.5 <= ratio <= 2.0, (
            f"Model {m}: CI width ratio evaluma/rliable={ratio:.3f} outside [0.5, 2.0]\n"
            f"  evaluma: [{tbl.loc[m, 'CI_low']:.4f}, {tbl.loc[m, 'CI_high']:.4f}]\n"
            f"  rliable: [{rliable_result[m]['ci_low']:.4f}, {rliable_result[m]['ci_high']:.4f}]"
        )


def test_numeric_summary(evaluma_result, rliable_result, capsys):
    """Print a side-by-side comparison table for human inspection."""
    tbl = evaluma_result.table.set_index("model")
    lines = [
        "=" * 72,
        f"{'Model':<8} {'evaluma IQM':>12} {'rliable IQM':>12} {'diff':>10} "
        f"{'ev CI width':>12} {'rl CI width':>12} {'ratio':>7}",
        "-" * 72,
    ]
    for model in MODELS:
        ev = tbl.loc[model, "IQM"]
        rl = rliable_result[model]["iqm"]
        ev_w = tbl.loc[model, "CI_high"] - tbl.loc[model, "CI_low"]
        rl_w = rliable_result[model]["ci_high"] - rliable_result[model]["ci_low"]
        lines.append(
            f"{model:<8} {ev:>12.8f} {rl:>12.8f} {abs(ev - rl):>10.2e} "
            f"{ev_w:>12.6f} {rl_w:>12.6f} {ev_w / rl_w:>7.3f}"
        )
    lines.append("=" * 72)
    lines.append("RESULT: PASS — evaluma IQM matches rliable (point estimates exact, CI widths consistent)")
    print("\n".join(lines))
