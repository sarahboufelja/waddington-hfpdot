"""Runs two samplers on the SAME 20x20 HFPD-OT problem (unbalanced / positive orthant or balanced / simplex):

  (A) vanilla MALA on the full II*JJ = 400 plan entries  -- the baseline that degenerates;
  (B) low-rank MALA r=2 with balanced gauge: on the m = r(II+JJ) = 80 free coords theta, and
    pi = exp(U.V^T - C/eps) or pi = softmax(U.V^T - C/eps) (the identifiable subspace of the latent space).

Both warm-start from perturbed versions of the Sinkhorn plan. Diagnostics are reported on a common,
gauge-invariant footing: eBFMI from the (param-invariant) energy, and R-hat / ESS on
the reconstructed plan pi (NOT on theta, whose r^2 GL(r) directions are non-identifiable).

Run from the repo root:  python scripts/run_lowrank_bridge.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from typing import Dict, Literal
from supports import LowRank
import numpy as np
import jax
import jax.numpy as jnp
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

II = JJ = 20
EPS = 0.05
RANK = 2
COMMON = dict(num_samples=5_000, num_burnin=1_000, warm_up_steps=600,
              num_parallel_chains=4, step_size=0.02, seed=0)

def build_prior(support: Literal["simplex", "positive_orthant"]="positive_orthant"):
    src = jnp.linspace(0.0, 1.0, II)
    tgt = jnp.linspace(0.0, 1.0, JJ)
    cost = (src[:, None] - tgt[None, :]) ** 2
    cost = jnp.asarray(cost / cost.max())
    mu_0 = jnp.exp(-0.5 * ((src - 0.35) / 0.15) ** 2)
    nu_0 = jnp.exp(-0.5 * ((tgt - 0.65) / 0.18) ** 2)
    mu_0 = mu_0 / mu_0.sum()
    if support == "simplex":
        mu_0 = mu_0 / mu_0.sum()  # balanced source
        nu_0 = nu_0 / nu_0.sum()  # balanced target
    else:
        nu_0 = 1.3 * nu_0 / nu_0.sum()  # 30% proliferation (unbalanced)
    prior = HFPDOTHyperprior(
        mu_0=jnp.asarray(mu_0),
        nu_0=jnp.asarray(nu_0),
        lambda_1=1.0, lambda_2=1.0, lambda_I_1=0.01, lambda_I_2=0.01,
        cost_fn=cost, epsilon=EPS, support=support,
    )
    return prior


def run(tag: str, sampler: MetropolisAdjustedLangevinSampler) -> Dict:
    """Drive the chains; report eBFMI (on the latent energy) and R-hat/ESS on pi.

    eBFMI and R-hat/ESS live in different spaces and must NOT be collapsed onto one
    array: eBFMI uses the sampler's own latent log-prob, so it is computed on the
    per-chain LATENT samples; R-hat/ESS are reported on the gauge-invariant plan pi,
    reconstructed per-sample via ``recon`` (latent -> pi). Both keep the (C, N, .) axis.
    """
    final_states = sampler.sample()

    acc = np.asarray(final_states.num_accepted_samples)[:, -1, 0] / sampler.tot_num_samples

    rep = dict(
        acc=np.round(acc, 2).tolist(),
        ebfmi_min=round(float(final_states.diagnostics.ebfmi_min), 3),
        pi_rhat_max=round(float(final_states.diagnostics.bulk_rhat_max), 3),
        pi_rhat_med=round(float(np.nanmedian(np.asarray(final_states.diagnostics.bulk_rhat))), 3),
        pi_ess_min=round(float(final_states.diagnostics.ess_min), 1),
        pi_ess_med=round(float(np.nanmedian(np.asarray(final_states.diagnostics.ess))), 1),
    )
    print(f"  {tag}: {rep}")
    return rep


def main():
    results = {}
    for support in ("positive_orthant", "simplex"):
        regime = "unbalanced" if support == "positive_orthant" else "balanced"
        prior = build_prior(support)          # each regime gets its OWN prior (marginals + support)
        print(f"\n====== HFPD-OT bridge: {II}x{JJ} {regime}, eps={EPS} ======\n")

        # (A) vanilla MALA on the full 400-dim plan
        vanilla = MetropolisAdjustedLangevinSampler(
            target_log_prob_fn=prior.hyperprior_log_prob_fun,
            target_score_fn=prior.hyperprior_score_fun,
            shape=II * JJ, support=support,
            initial_plan=prior.sinkhorn_init(), **COMMON,
        )
        print("(A) vanilla MALA on the full 400-dim plan ...")
        results[f"vanilla_{regime}"] = run(f"vanilla_400d_{regime}", vanilla)

        # (B) low-rank r=2; recon = plan(unpack(theta)) (theta -> pi).
        lp = LowRank(II, JJ, RANK, cost=prior.cost_fn, epsilon=EPS, support=support,)
        theta0 = lp.to_unconstrained(prior.sinkhorn_init())
        warm_err = float(jnp.linalg.norm(lp.to_constrained(theta0).reshape(-1) - prior.sinkhorn_init().reshape(-1)))
        print(f"(B) Low-rank r={RANK} on the {lp.num_free}-dim coords ...")
        print(f"    warm-start ||theta0||={float(jnp.linalg.norm(theta0)):.2f}  "
              f"||recon(theta0)-sinkhorn||={warm_err:.2e}")

        # Gauge-breaking ridge: the simplex low-rank has an UNBOUNDED softmax-shift gauge
        # (softmax(ell+c1)=softmax(ell)) that freezes pi; a ridge on ||theta|| pins it. The
        # orthant has no such runaway (exp rescales mass, which the hyperprior constrains) -> 0.
        ridge = 0.1 if support == "simplex" else 0.0

        def lowrank_sampler(spread, lp=lp, ridge=ridge):
            special_kwargs = dict(II=II, JJ=JJ, rank=RANK, cost=prior.cost_fn, epsilon=EPS, ridge=ridge)
            return MetropolisAdjustedLangevinSampler(
                target_log_prob_fn=prior.hyperprior_log_prob_fun,
                target_score_fn=prior.hyperprior_score_fun,
                shape=lp.num_free,
                sampling_strategy="low_rank",
                support=support,
                initial_plan=prior.sinkhorn_init(),
                warm_start_sigma=spread, **COMMON, **special_kwargs
            )

        # Standard (over-dispersed) spread = the honest comparison; 
        # Tight spread = the DISCRIMINATING probe (pseudo-non-convergence vs genuine multimodality),
        # not the final config.
        results[f"lowrank_std_{regime}"] = run(f"lowrank_std(0.1,0.3)_{regime}", lowrank_sampler((0.1, 0.3)))
        results[f"lowrank_tight_{regime}"] = run(f"lowrank_tight(0.02,0.08)_{regime}", lowrank_sampler((0.02, 0.08)))

    print("\nVERDICT (degeneracy = high R-hat / low ESS / low eBFMI):")
    for tag, rep in results.items():
        print(f"  {tag:28s}: eBFMI {rep['ebfmi_min']}  pi-Rhat(max/med) {rep['pi_rhat_max']}/{rep['pi_rhat_med']}"
              f"  pi-ESS(min/med) {rep['pi_ess_min']}/{rep['pi_ess_med']}")


if __name__ == "__main__":
    main()
