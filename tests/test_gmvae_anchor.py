"""Stage B supervised anchor loss.

The anchor is soft multi-label cross-entropy on the labelled cells only, with the class weight folded
into the class sum so ambiguous cells weight each of their populations by that population's own w_c.
These tests pin the reduction (mean over labelled), the ambiguity handling, and that gradient reaches
the responsibilities (hence the encoder and the GMM prior).
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from gmvae.losses import LossFunctions  # noqa: E402

L = LossFunctions()
K = 3


def _log_gamma(gamma):
    return torch.log(torch.as_tensor(gamma, dtype=torch.float32))


# ---- the value on hand cases --------------------------------------------------------------------

def test_one_hot_targets_uniform_weights_is_mean_neg_log_gamma():
    """One-hot labels, w=1: L = mean over labelled of -log gamma at the true class."""
    gamma = torch.tensor([[0.7, 0.2, 0.1],
                          [0.1, 0.8, 0.1]])
    targets = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
    got = L.anchor_loss(torch.log(gamma), targets, torch.ones(K))
    expected = -0.5 * (math.log(0.7) + math.log(0.8))
    assert torch.allclose(got, torch.tensor(expected), atol=1e-5)


def test_unlabelled_cells_are_excluded_from_the_mean():
    """A zero-target (unlabelled) row must not enter the loss or its denominator."""
    gamma = torch.tensor([[0.7, 0.2, 0.1],
                          [0.33, 0.33, 0.34],   # unlabelled
                          [0.1, 0.8, 0.1]])
    targets = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],     # unlabelled -> excluded
                            [0.0, 1.0, 0.0]])
    got = L.anchor_loss(torch.log(gamma), targets, torch.ones(K))
    expected = -0.5 * (math.log(0.7) + math.log(0.8))     # denominator is 2, not 3
    assert torch.allclose(got, torch.tensor(expected), atol=1e-5)


def test_empty_batch_returns_zero():
    gamma = torch.full((4, K), 1.0 / K)
    targets = torch.zeros(4, K)
    got = L.anchor_loss(torch.log(gamma), targets, torch.ones(K))
    assert got.item() == 0.0


def test_class_weight_scales_each_class_contribution():
    """w_c multiplies the per-class term; upweighting a rare class raises its cell's loss."""
    gamma = torch.tensor([[0.1, 0.9]])                    # cell is labelled class 0, poorly predicted
    targets = torch.tensor([[1.0, 0.0]])
    base = L.anchor_loss(torch.log(gamma), targets, torch.tensor([1.0, 1.0]))
    up = L.anchor_loss(torch.log(gamma), targets, torch.tensor([5.0, 1.0]))
    assert torch.allclose(up, 5.0 * base, atol=1e-5)


def test_ambiguous_cell_weights_each_population_by_its_own_wc():
    """A 0.5/0.5 cell: each half is scaled by that population's own class weight, not one per-cell w."""
    gamma = torch.tensor([[0.5, 0.5]])
    targets = torch.tensor([[0.5, 0.5]])
    w = torch.tensor([4.0, 1.0])
    got = L.anchor_loss(torch.log(gamma), targets, w)
    expected = -(4.0 * 0.5 * math.log(0.5) + 1.0 * 0.5 * math.log(0.5))
    assert torch.allclose(got, torch.tensor(expected), atol=1e-5)


# ---- it is a real, differentiable objective ------------------------------------------------------

def test_gradient_flows_to_the_responsibilities():
    log_gamma = torch.log_softmax(torch.randn(6, K, requires_grad=True), dim=-1)
    log_gamma.retain_grad()
    targets = torch.zeros(6, K)
    targets[torch.arange(6), torch.randint(0, K, (6,))] = 1.0   # random one-hot labels
    L.anchor_loss(log_gamma, targets, torch.ones(K)).backward()
    assert log_gamma.grad is not None and float((log_gamma.grad ** 2).sum()) > 0


def test_loss_is_lower_when_predictions_match_labels():
    targets = torch.eye(K)                                  # one cell per class, one-hot
    good = torch.log(torch.eye(K) * 0.9 + 0.05)             # confident and correct
    bad = torch.log(torch.ones(K, K) / K)                  # uniform (uninformative)
    assert L.anchor_loss(good, targets, torch.ones(K)) < L.anchor_loss(bad, targets, torch.ones(K))
