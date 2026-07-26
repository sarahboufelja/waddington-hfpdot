"""Stage A (plain-VAE pretraining) contract tests.

Stage A builds a structured latent BEFORE the GMM prior is seeded: it trains the encoder + decoder
(+ per-gene dispersion) under a standard-normal prior, and must leave the GMM prior (mu_c, logvar_c,
pi_logits) completely untouched. The load-bearing test is exactly that isolation -- if a pretraining
step moved the prior, SEED would be fitting to a prior that had already drifted.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from gmvae.networks import GMVAENet  # noqa: E402

X_DIM, LATENT, HIDDEN, K, BATCH = 60, 8, 32, 4, 16


def _model(seed=0):
    torch.manual_seed(seed)
    return GMVAENet(x_dim=X_DIM, num_clusters=K, latent_dim=LATENT, hidden_dim=HIDDEN)


def _gmm_params(model):
    """The GMM prior parameters, by reference -- identified by identity, not name (the GMVAENet
    aliases shadow the cluster_prior.* names)."""
    cp = model.cluster_prior
    return {"mu_c": cp.mu_c, "logvar_c": cp.logvar_c, "pi_logits": cp.pi_logits}


def _counts(batch=BATCH, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rates = torch.rand(batch, X_DIM, generator=gen) * 3.0
    mask = torch.rand(batch, X_DIM, generator=gen) > 0.9
    return torch.poisson(rates * mask, generator=gen)


# ---- the loss is a real, finite, scalar VAE objective --------------------------------------------

def test_pretrain_loss_returns_finite_scalar_parts_that_sum():
    model = _model()
    total, recon, kl = model.pretrain_loss(_counts(), beta=1.0)
    for t in (total, recon, kl):
        assert torch.is_tensor(t) and t.ndim == 0 and torch.isfinite(t)
    assert torch.allclose(total, recon + kl, atol=1e-5)          # beta=1
    assert kl >= 0.0                                             # KL to N(0,I) is non-negative


def test_kl_matches_the_closed_form_standard_normal_value():
    """KL(N(mu, sigma^2) || N(0, I)) = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar), summed over dims,
    meaned over cells. Checked against a hand computation on the encoder's own outputs."""
    model = _model()
    gauss = model.inference_net.q_zx(_counts())
    mu, logvar = gauss.mu, gauss.logvar
    expected = (0.5 * (torch.exp(logvar) + mu ** 2 - 1.0 - logvar).sum(-1)).mean()
    zeros = torch.zeros_like(mu)
    got = model.losses.gaussian_kl_diag(mu, logvar, zeros, zeros).mean()
    assert torch.allclose(got, expected, atol=1e-5)


def test_beta_scales_only_the_kl_term():
    model = _model()
    counts = _counts()
    torch.manual_seed(1); t1, r1, k1 = model.pretrain_loss(counts, beta=1.0)   # seed -> same z
    torch.manual_seed(1); t2, r2, k2 = model.pretrain_loss(counts, beta=3.0)
    assert torch.allclose(r1, r2) and torch.allclose(k1, k2)     # same reparam sample -> same parts
    assert torch.allclose(t2, r2 + 3.0 * k2, atol=1e-5)


# ---- Stage A isolates the GMM prior (the load-bearing guarantee) ---------------------------------

def test_pretrain_parameters_exclude_the_gmm_prior():
    model = _model()
    stage_a = {id(p) for p in model.pretrain_parameters()}
    for name, p in _gmm_params(model).items():
        assert id(p) not in stage_a, f"{name} must not be in the Stage A parameter set"
    assert len(list(model.pretrain_parameters())) > 0           # non-empty (encoder+decoder+disp)


def test_pretrain_step_updates_encoder_decoder_but_not_the_gmm_prior():
    """A pretraining step must move the encoder/decoder and leave mu_c/logvar_c/pi_logits identical."""
    model = _model()
    gmm = _gmm_params(model)
    gmm_ids = {id(p) for p in gmm.values()}
    before_gmm = {k: p.detach().clone() for k, p in gmm.items()}
    before_all = {n: p.detach().clone() for n, p in model.named_parameters()}

    opt = torch.optim.Adam(model.pretrain_parameters(), lr=1e-2)
    opt.zero_grad()
    model.pretrain_loss(_counts(), beta=1.0)[0].backward()
    opt.step()

    for k, p in gmm.items():                                     # prior frozen, and never got a grad
        assert torch.equal(p, before_gmm[k]), f"{k} changed during Stage A"
        assert p.grad is None, f"{k} received a gradient in Stage A"

    moved = [n for n, p in model.named_parameters()             # encoder/decoder did move
             if id(p) not in gmm_ids and not torch.equal(p, before_all[n])]
    assert moved, "no encoder/decoder parameter changed -- Stage A did nothing"


def test_a_few_pretrain_steps_reduce_the_loss():
    model = _model()
    counts = _counts()
    opt = torch.optim.Adam(model.pretrain_parameters(), lr=1e-3)
    first = float(model.pretrain_loss(counts)[0].detach())
    for _ in range(30):
        opt.zero_grad()
        loss = model.pretrain_loss(counts)[0]
        loss.backward()
        opt.step()
    assert float(loss.detach()) < first
