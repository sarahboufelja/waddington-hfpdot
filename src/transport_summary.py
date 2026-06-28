"""Pure (matplotlib-free) reductions for visualizing posterior transport plans.

These turn a posterior *sample* of HFPD-OT / OT plans into the quantities a plot
needs -- a mean plan, a per-entry std, marginal credible bands -- and a semantic
ordering of the source/target axes so the plan can be read as block-to-block
mass propagation. Kept free of matplotlib so they stay unit-testable; the drawing
lives in ``sampler_viz.plot_transport_plan``.

  - transport_plan_summary(...)  -> mean/std plan + marginal credible bands
  - dendrogram_order(...)        -> hierarchical-clustering leaf order
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class TransportPlanSummary:
    mean_plan: np.ndarray
    std_plan: np.ndarray
    mean_first_marginal: np.ndarray
    mean_second_marginal: np.ndarray
    lo_first_marginal: np.ndarray
    hi_first_marginal: np.ndarray
    lo_second_marginal: np.ndarray
    hi_second_marginal: np.ndarray

def dendrogram_order(
    features, method: str = "ward", metric: str = "euclidean", optimal_ordering: bool = False
) -> np.ndarray:
    """Leaf order from hierarchical clustering: places similar items adjacent.

    Reordering the source (rows) and/or target (cols) axes of a transport plan by
    such an order collapses similar items into contiguous blocks, so the heatmap
    reads as mass flowing from input clusters to target nodes.

    Args:
        features: ``(n, k)`` array, one row per item (source or target) described by
            ``k`` features -- e.g. GMVAE latent coordinates / expression profiles. If
            no external features are available, pass the plan's own rows (or columns)
            to cluster items by their transport behavior. A 1-D ``(n,)`` coordinate
            (e.g. pseudotime) is accepted and treated as a single feature.
        method: scipy linkage method ("ward", "average", "complete", ...).
        metric: pairwise distance metric. Ignored when ``method="ward"`` (which is
            defined for Euclidean distances only).
        optimal_ordering: if True, reorder each merge to minimize the
            distance between adjacent leaves -- yields the most readable heatmap
            ordering (monotonic for a 1-D coordinate). O(n^2); disable for very
            large n if it becomes a bottleneck.

    Returns:
        ``(n,)`` int permutation of ``range(n)`` giving the dendrogram leaf order.
        Apply as ``plan[order]`` (rows) or ``plan[:, order]`` (cols). For ``n < 2``
        the identity order is returned.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list

    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    n = features.shape[0]
    if n < 2:
        return np.arange(n, dtype=int)

    # Ward is only defined for Euclidean distances; scipy rejects an explicit metric.
    if method == "ward":
        linkage_matrix = linkage(features, method="ward", optimal_ordering=optimal_ordering)
    else:
        linkage_matrix = linkage(features, method=method, metric=metric, optimal_ordering=optimal_ordering)

    return leaves_list(linkage_matrix).astype(int)

def transport_plan_summary(plans: np.ndarray) -> TransportPlanSummary:
    """
    Summarizes the posterior transport plan by its mean, per-entry std and marginal credible intervals
    
    Args:
        plans: array of shape (num_samples, n_source, n_target) representing the posterior
    
    Returns:
        TransportPlanSummary: dataclass containing the mean plan, std plan, mean first and second
    """
    mean_plan = np.mean(plans, axis=0)
    std_plan = np.std(plans, axis=0)

    # Credible intervals around the marginals
    first_marginals = np.sum(plans, axis=2)
    second_marginals = np.sum(plans, axis=1)

    mean_first_marginal = np.mean(first_marginals, axis=0)
    mean_second_marginal = np.mean(second_marginals, axis=0)
    lo_first_marginal = np.percentile(first_marginals, 2.5, axis=0)
    hi_first_marginal = np.percentile(first_marginals, 97.5, axis=0)
    lo_second_marginal = np.percentile(second_marginals, 2.5, axis=0)
    hi_second_marginal = np.percentile(second_marginals, 97.5, axis=0)

    return TransportPlanSummary(
        mean_plan=mean_plan,
        std_plan=std_plan,
        mean_first_marginal=mean_first_marginal,
        mean_second_marginal=mean_second_marginal,
        lo_first_marginal=lo_first_marginal,
        hi_first_marginal=hi_first_marginal,
        lo_second_marginal=lo_second_marginal,
        hi_second_marginal=hi_second_marginal
    )