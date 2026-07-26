"""Contract tests for the marginalised-VaDE GMVAE: does it run, produce finite losses, and can it
learn?

Not a test of what the model represents -- only that the machinery is wired up. These exist because
the model had never been executed: the losses were stubbed to zero, so nothing exercised the forward
path, and several wiring faults sat undetected in it.

The scale check is the load-bearing one. The ELBO is a sum over each variable's dimensions and a mean
over cells; if the reconstruction term is averaged over genes as well, it enters ~n_genes times too
weakly, the KL terms dominate, and the posterior collapses to the prior -- which would leave every
cell with the same latent and no usable uncertainty.

The contract is docs/gmvae_model.md: one posterior q(z,c|x) yielding mean_latent, var_latent (the
latent Gaussian) and prob_cat = gamma (the soft membership = responsibilities). No Gumbel-softmax, no
temperature, no separate q(c|x) head.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from gmvae.networks import GMVAENet  # noqa: E402

X_DIM, LATENT, HIDDEN, K, BATCH = 60, 8, 32, 4, 16


def _model(**kw):
    torch.manual_seed(0)
    params = dict(x_dim=X_DIM, num_clusters=K, latent_dim=LATENT, hidden_dim=HIDDEN)
    params.update(kw)
    return GMVAENet(**params)


def _counts(batch=BATCH, sparsity=0.9, seed=0):
    """Sparse non-negative integer counts, in the shape single-cell data actually arrives."""
    gen = torch.Generator().manual_seed(seed)
    rates = torch.rand(batch, X_DIM, generator=gen) * 3.0
    mask = torch.rand(batch, X_DIM, generator=gen) > sparsity
    return torch.poisson(rates * mask, generator=gen)


# ---- it runs at all -----------------------------------------------------------------------------

def test_forward_runs_and_returns_the_expected_fields():
    out = _model()(_counts())
    for field in ("mean_latent", "logvar_latent", "latent_sample", "log_gamma_c",
                  "var_latent", "prob_cat", "x_recon"):
        assert hasattr(out, field), f"missing {field}"


def test_latent_and_categorical_shapes():
    out = _model()(_counts())
    assert out.mean_latent.shape == (BATCH, LATENT)
    assert out.var_latent.shape == (BATCH, LATENT)
    assert out.latent_sample.shape == (BATCH, LATENT)
    assert out.prob_cat.shape == (BATCH, K)


def test_variances_are_positive_and_probabilities_normalised():
    """These feed the radius and the soft membership directly, so their validity is a contract."""
    out = _model()(_counts())
    assert torch.all(out.var_latent > 0), "a non-positive variance breaks every downstream KL"
    assert torch.allclose(out.prob_cat.sum(dim=-1), torch.ones(BATCH), atol=1e-5)
    assert torch.all(out.prob_cat >= 0)


# ---- the losses are real numbers, not stubs ------------------------------------------------------

@pytest.mark.parametrize("field", ["recon_loss", "gaussian_loss", "categorical_loss", "total_loss"])
def test_losses_are_finite_scalars(field):
    out = _model()(_counts())
    value = getattr(out, field)
    assert torch.is_tensor(value), f"{field} is {type(value).__name__}, not a tensor"
    assert value.ndim == 0, f"{field} is not a scalar (shape {tuple(value.shape)}); backward needs it"
    assert torch.isfinite(value).all(), f"{field} is not finite"


def test_total_loss_is_reconstruction_plus_beta_weighted_kl():
    out = _model(beta=1.0)(_counts())
    assert torch.allclose(out.total_loss,
                          out.recon_loss + out.gaussian_loss + out.categorical_loss, atol=1e-5)


def test_beta_scales_only_the_kl_terms():
    """beta is the explicit KL weight (spec decision 3): it multiplies (2)+(3)+(4), never (1)."""
    counts = _counts()
    a = _model(beta=1.0)(counts)
    b = _model(beta=2.0)(counts)
    # same seed -> same recon and same KL terms; only their combination changes
    assert torch.allclose(a.recon_loss, b.recon_loss, atol=1e-5)
    assert torch.allclose(b.total_loss,
                          b.recon_loss + 2.0 * (b.gaussian_loss + b.categorical_loss), atol=1e-5)


def test_loss_terms_are_on_a_comparable_scale():
    """The ELBO sums over each variable's dimensions and averages over cells. If the reconstruction
    term is also averaged over genes it enters ~n_genes times too weakly and the KL terms dominate,
    driving the posterior to the prior. Orders of magnitude apart is the symptom."""
    out = _model()(_counts())
    recon = abs(float(out.recon_loss.detach()))
    kl = abs(float(out.gaussian_loss.detach())) + abs(float(out.categorical_loss.detach()))
    assert kl > 0, "KL terms vanished -- nothing constrains the latent"
    ratio = recon / kl
    assert 1e-2 < ratio < 1e4, (
        f"reconstruction/KL = {ratio:.2e}; the terms are not commensurate. Check that the "
        f"reconstruction sums over genes and only averages over cells.")


# ---- it can actually learn -----------------------------------------------------------------------

def test_backward_reaches_every_parameter():
    """A zero or missing gradient means part of the network is detached from the objective -- the
    responsibilities in particular must keep the GMM prior (mu_c, logvar_c, pi_logits) in the graph."""
    model = _model()
    model(_counts()).total_loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing[:5]}"
    total = sum(float((p.grad ** 2).sum()) for _, p in model.named_parameters() if p.grad is not None)
    assert total > 0, "all gradients are exactly zero"


def test_the_gmm_prior_receives_gradient():
    """The single-source guarantee needs the prior to train, not just the encoder/decoder."""
    model = _model()
    model(_counts()).total_loss.backward()
    for name in ("mu_c", "logvar_c", "pi_logits"):
        p = getattr(model.cluster_prior, name)
        assert p.grad is not None and float((p.grad ** 2).sum()) > 0, f"{name} got no gradient"


def test_a_few_steps_reduce_the_loss_on_a_fixed_batch():
    """The weakest possible learning check: the model must be able to fit one batch."""
    model = _model()
    counts = _counts()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = float(model(counts).total_loss.detach())
    for _ in range(30):
        opt.zero_grad()
        loss = model(counts).total_loss
        loss.backward()
        opt.step()
    last = float(loss.detach())
    assert last < first, f"loss did not decrease ({first:.3f} -> {last:.3f})"


# ---- the reconstruction is a rate per gene -------------------------------------------------------

def test_generative_arm_emits_one_value_per_gene():
    """x is a vector of gene counts modelled as conditionally independent, so the decoder must emit
    one (log-)rate per gene -- not a single scalar rate for the whole cell."""
    out = _model()(_counts())
    assert out.x_recon.shape == (BATCH, X_DIM)


def test_reconstruction_head_is_unconstrained_log_rates():
    """With PoissonNLLLoss(log_input=True) the head emits log-rates, which may be negative; a
    non-negative output would mean an activation is still clamping them."""
    out = _model()(_counts(seed=3))
    recon = out.x_recon.detach()
    assert torch.isfinite(recon).all()
    assert float(recon.min()) < 0.0, "log-rates never go negative -- is an activation clamping them?"
