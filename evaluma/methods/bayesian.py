from itertools import combinations

import baycomp
import pandas as pd

from evaluma.results import BayesianResult


def compute_bayesian(
    scores_matrix: pd.DataFrame,
    *,
    rope=0.01,
    reference=None,
    pairs=None,
    random_state=None,
) -> BayesianResult:
    """Compute pairwise Bayesian comparisons using a signed-rank test.

    For each pair, :func:`baycomp.two_on_multiple` returns the posterior
    probability that model A is better, that they are practically
    equivalent (within ``rope``), and that model B is better.

    Args:
        scores_matrix: Normalized model × dataset score matrix.
        rope: Region of practical equivalence half-width **in normalized
            score space (0–1)**. Differences smaller than ``rope`` are
            treated as practically equivalent.
        reference: If provided, only compare every other model against this
            reference model.
        pairs: Explicit list of ``(model_a, model_b)`` pairs to test.
            Overrides ``reference``.
        random_state: Seed forwarded to baycomp.

    Returns:
        BayesianResult: Result with ``.table`` containing columns
        ``model_a``, ``model_b``, ``p_a_better``, ``p_equiv``,
        ``p_b_better``.
    """
    models = scores_matrix.index.tolist()

    if pairs is not None:
        pair_list = list(pairs)
    elif reference is not None:
        pair_list = [(reference, m) for m in models if m != reference]
    else:
        pair_list = list(combinations(models, 2))

    rows = []
    for model_a, model_b in pair_list:
        vec_a = scores_matrix.loc[model_a].values.astype(float)
        vec_b = scores_matrix.loc[model_b].values.astype(float)
        p_a, p_eq, p_b = baycomp.two_on_multiple(
            vec_a, vec_b, rope=rope, random_state=random_state
        )
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "p_a_better": p_a,
                "p_equiv": p_eq,
                "p_b_better": p_b,
            }
        )

    return BayesianResult(pd.DataFrame(rows), reference=reference)
