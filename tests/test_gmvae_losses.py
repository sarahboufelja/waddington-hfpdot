"""Math tests for the GMVAE reconstruction likelihood (negative binomial).

The NB replaces Poisson so the decoder can model scRNA-seq overdispersion (Var > mean). The
parameterisation is NB2: the decoder head emits the log-mean (prediction = log m(z)) and a separate
per-gene dispersion r (torch's total_count) controls the variance, Var = mean + mean^2 / r. Poisson is
the r -> inf limit. These tests pin exactly those three facts plus the reduction and gradient flow.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from gmvae.losses import LossFunctions  # noqa: E402


def _nb(prediction, total_count):
    """The distribution NBNLLLoss builds internally, for checking its moments."""
    logits = prediction - torch.log(total_count)
    return torch.distributions.NegativeBinomial(total_count=total_count, logits=logits)


# ---- the parameterisation: mean = exp(prediction), independent of the dispersion -----------------

@pytest.mark.parametrize("r", [0.5, 2.0, 1e3, 1e6])
def test_mean_is_exp_prediction_regardless_of_dispersion(r):
    """prediction is the log-mean; total_count must not shift the mean, only the variance."""
    prediction = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    total_count = torch.full_like(prediction, r)
    dist = _nb(prediction, total_count)
    assert torch.allclose(dist.mean, torch.exp(prediction), rtol=1e-4, atol=1e-4)


def test_variance_is_overdispersed_nb2():
    """Var = mean + mean^2 / r, strictly above the Poisson variance (= mean)."""
    prediction = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    r = torch.full_like(prediction, 3.0)
    dist = _nb(prediction, r)
    mean = torch.exp(prediction)
    assert torch.allclose(dist.variance, mean + mean.pow(2) / r, rtol=1e-4, atol=1e-4)
    assert torch.all(dist.variance > mean), "NB must be over-dispersed relative to Poisson"


# ---- Poisson is the large-dispersion limit -------------------------------------------------------

def test_reduces_to_poisson_as_dispersion_grows():
    """As r -> inf the NB NLL converges to the *exact* Poisson NLL with the same mean. The gap must
    shrink monotonically in r -- the sense in which NB strictly generalises Poisson. The reference is
    torch.distributions.Poisson (exact lgamma), not PoissonNLLLoss, whose default full=False drops the
    log(x!) normaliser that NB.log_prob keeps. Float64 throughout: at r>~1e6 float32 loses the limit
    to catastrophic cancellation in lgamma(r+x) - lgamma(r)."""
    gen = torch.Generator().manual_seed(0)
    prediction = (torch.randn(64, 20, generator=gen) * 0.5).double()   # log-rates around 1
    real = torch.poisson(torch.exp(prediction), generator=gen)
    poisson = -torch.distributions.Poisson(torch.exp(prediction)).log_prob(real)

    gaps = []
    for r in (1e1, 1e2, 1e4, 1e6):
        total_count = torch.full_like(prediction, r)
        nb = LossFunctions.NBNLLLoss(prediction, real, total_count)
        gaps.append(float((nb - poisson).abs().max()))
    assert gaps == sorted(gaps, reverse=True), f"gap not monotone in r: {gaps}"
    assert gaps[-1] < 1e-3, f"NB did not converge to Poisson at r=1e6 (max gap {gaps[-1]:.2e})"


# ---- reduction and gradient ----------------------------------------------------------------------

def test_reconstruction_loss_sums_genes_means_batch_to_a_scalar():
    """Sum over the D genes (joint log-likelihood), mean over the batch (MC over cells) -> scalar."""
    batch, genes = 16, 30
    gen = torch.Generator().manual_seed(1)
    prediction = torch.randn(batch, genes, generator=gen) * 0.5
    real = torch.poisson(torch.exp(prediction), generator=gen)
    theta = torch.full((genes,), 5.0)                          # per-gene dispersion, broadcasts
    loss = LossFunctions().reconstruction_loss(real, prediction, theta)
    assert loss.ndim == 0
    assert torch.isfinite(loss) and loss > 0                   # a NLL is non-negative here

    # matches an explicit sum-genes / mean-batch reduction of the per-element NLL
    per_elem = LossFunctions.NBNLLLoss(prediction, real, theta)
    assert torch.allclose(loss, per_elem.sum(dim=-1).mean(), atol=1e-5)


def test_gradient_flows_to_prediction_and_dispersion():
    """Both the decoder head (prediction) and the learnable dispersion must receive gradient."""
    gen = torch.Generator().manual_seed(2)
    prediction = (torch.randn(8, 12, generator=gen) * 0.5).requires_grad_(True)
    real = torch.poisson(torch.exp(prediction.detach()), generator=gen)
    raw_theta = torch.zeros(12, requires_grad=True)
    theta = torch.nn.functional.softplus(raw_theta)            # positive by construction
    LossFunctions().reconstruction_loss(real, prediction, theta).backward()
    assert prediction.grad is not None and float((prediction.grad ** 2).sum()) > 0
    assert raw_theta.grad is not None and float((raw_theta.grad ** 2).sum()) > 0
