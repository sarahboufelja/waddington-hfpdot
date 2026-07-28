"""The align_populations injection into build_targets and seed_prior.

The existing driver tests feed memberships whose population axis is already canonical, so the align
step is a no-op there. These feed the cases that actually exercise it: per-day memberships in
different column orders, a day missing a canonical fate, and a day carrying a fate absent from the
canonical axis (which must fail rather than silently drop labelled mass). The invariant under test is
that a target/seed column indexes the same fate regardless of how each day happened to order its
columns.
"""
import os
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from gmvae.networks import GMVAENet                                    # noqa: E402
from gmvae.embedder import VaDEEmbedder                               # noqa: E402
from gmvae.training import build_targets, seed_prior, _gmm_from_arrays  # noqa: E402
from wadd_data_ingest import CellAxis, ExpressionMatrix, Membership   # noqa: E402

N_GENES = 12
CANON = ["A", "B", "C"]


def _exp(ids, day, seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.poisson(1.0, size=(len(ids), N_GENES)).astype(np.float32)
    cells = CellAxis(ids=list(ids), day=day)
    return ExpressionMatrix(sparse.csr_matrix(dense), cells, [f"g{j}" for j in range(N_GENES)])


# ---- build_targets aligns per-day column axes ----------------------------------------------------

def test_permuted_and_subset_axes_land_on_the_canonical_columns():
    """Day 9's membership is columns [B,A,C]; day 10's is the subset [A,C]. build_targets must place
    every cell's mass under the right canonical fate."""
    exp9 = _exp(["c1", "c2"], 9.0)
    memb9 = Membership([[0.0, 1.0, 0.0],       # c1 -> A (index 1 in [B,A,C])
                        [0.0, 0.0, 1.0]],      # c2 -> C
                       exp9.cells, ["B", "A", "C"])
    exp10 = _exp(["c3", "c4"], 10.0)
    memb10 = Membership([[1.0, 0.0],           # c3 -> A (in [A,C])
                         [0.0, 1.0]],          # c4 -> C
                        exp10.cells, ["A", "C"])

    t = build_targets([exp9, exp10], [memb9, memb10], CANON)
    assert t.shape == (4, 3)
    assert np.allclose(t, [[1, 0, 0],          # c1 A
                           [0, 0, 1],          # c2 C
                           [1, 0, 0],          # c3 A
                           [0, 0, 1]])         # c4 C  (B column is all zeros: absent both days)


def test_build_targets_matches_pre_aligned_memberships():
    """Aligning inside build_targets must equal aligning the memberships first by hand."""
    exp9 = _exp(["c1", "c2"], 9.0)
    memb9 = Membership([[0.4, 0.6, 0.0], [0.0, 0.0, 1.0]], exp9.cells, ["B", "A", "C"])
    exp10 = _exp(["c3"], 10.0)
    memb10 = Membership([[1.0, 0.0]], exp10.cells, ["A", "C"])

    via_injection = build_targets([exp9, exp10], [memb9, memb10], CANON)
    pre = [m.align_populations(CANON) for m in (memb9, memb10)]
    via_hand = build_targets([exp9, exp10], pre, CANON)
    assert np.allclose(via_injection, via_hand)


def test_build_targets_rejects_a_fate_outside_the_canonical_axis():
    exp = _exp(["c1"], 9.0)
    memb = Membership([[0.5, 0.5]], exp.cells, ["A", "Z"])   # Z not in CANON
    with pytest.raises(ValueError, match="absent from canonical_names"):
        build_targets([exp], [memb], CANON)


# ---- seed_prior aligns per-day column axes -------------------------------------------------------

def test_seed_prior_is_invariant_to_per_day_column_order():
    """Two days with the SAME cell->fate content but different column orders must seed the identical
    prior -- reproduce the reduction independently from hand-aligned memberships and compare."""
    torch.manual_seed(0)
    model = GMVAENet(x_dim=N_GENES, num_clusters=3, latent_dim=4, hidden_dim=16)

    exp9 = _exp(["c1", "c2", "c3"], 9.0, seed=1)
    memb9 = Membership([[0.0, 1.0, 0.0],       # c1 -> A  in [B,A,C]
                        [1.0, 0.0, 0.0],       # c2 -> B
                        [0.0, 0.0, 1.0]],      # c3 -> C
                       exp9.cells, ["B", "A", "C"])
    exp10 = _exp(["c4", "c5"], 10.0, seed=2)
    memb10 = Membership([[1.0, 0.0],           # c4 -> A  in [A,C]
                         [0.0, 1.0]],          # c5 -> C
                        exp10.cells, ["A", "C"])

    seed_prior(model, [exp9, exp10], [memb9, memb10], CANON, temperature=0.5, device="cpu")

    # Independent reduction: embed with the same (seed-run) encoder, hand-align to CANON, aggregate.
    embedder = VaDEEmbedder(model, device="cpu")
    mus, vars_, Ws = [], [], []
    for exp, memb in [(exp9, memb9), (exp10, memb10)]:
        post = embedder.embed(exp)
        mus.append(post.means); vars_.append(post.variances)
        Ws.append(memb.align_populations(CANON).for_cells(post.cells))
    means, vars_c, w = _gmm_from_arrays(np.vstack(Ws), np.vstack(mus), np.vstack(vars_), 0.5)

    assert np.allclose(model.cluster_prior.mu_c.detach().numpy(), means, atol=1e-5)
    assert np.allclose(torch.exp(model.cluster_prior.logvar_c).detach().numpy(), vars_c, atol=1e-5)
    assert np.allclose(torch.softmax(model.cluster_prior.pi_logits, -1).detach().numpy(),
                       w / w.sum(), atol=1e-5)


def test_seed_prior_rejects_a_fate_outside_the_canonical_axis():
    torch.manual_seed(0)
    model = GMVAENet(x_dim=N_GENES, num_clusters=3, latent_dim=4, hidden_dim=16)
    exp = _exp(["c1"], 9.0)
    memb = Membership([[0.5, 0.5]], exp.cells, ["A", "Z"])
    with pytest.raises(ValueError, match="absent from canonical_names"):
        seed_prior(model, [exp], [memb], CANON, device="cpu")
