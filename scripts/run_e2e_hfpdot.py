"""End-to-end HFPD-OT pipeline demo.

Runs the full block on a realistic 20x20 transport-plan problem:
    prior init -> MALA sampling (positive orthant) -> MCMC diagnostics -> plots,

and writes the artifacts under ``assets/images/<run_tag>/`` (run_tag = timestamp)
as evidence that the whole pipeline works together.

Run from the repo root:
    python scripts/run_e2e_hfpdot.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Deterministic CPU run with one device per chain (set before importing jax).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior
from sampler_viz import plot_transport_plan
from transport_summary import dendrogram_order


def build_prior(support, II=20, JJ=20, epsilon=0.05):
    """A structured 20x20 problem. The source/target supports are points on [0, 1], and the
    nominal marginals mu_0, nu_0 are bell-shaped masses over those points (positive by
    construction). For the balanced (simplex) regime they are normalized to probability
    vectors; for the unbalanced (orthant) regime they are kept as positive masses with a
    deliberate source/target imbalance, mimicking proliferation."""
    src = np.linspace(0.0, 1.0, II)
    tgt = np.linspace(0.0, 1.0, JJ)
    cost = (src[:, None] - tgt[None, :]) ** 2          # (II, JJ) squared euclidean
    cost = jnp.asarray(cost / cost.max())

    mu_0 = np.exp(-0.5 * ((src - 0.35) / 0.15) ** 2)   # source bell
    nu_0 = np.exp(-0.5 * ((tgt - 0.65) / 0.18) ** 2)   # shifted target bell
    if support == "simplex":
        mu_0, nu_0 = mu_0 / mu_0.sum(), nu_0 / nu_0.sum()         # probability vectors
    else:  # positive_orthant: positive masses; mass need not be conserved
        mu_0, nu_0 = mu_0 / mu_0.sum(), 1.3 * nu_0 / nu_0.sum()   # 30% proliferation
    mu_0, nu_0 = jnp.asarray(mu_0), jnp.asarray(nu_0)

    prior = HFPDOTHyperprior(
        mu_0=mu_0, nu_0=nu_0,
        lambda_1=1.0, lambda_2=1.0, lambda_I_1=0.01, lambda_I_2=0.01,
        cost_fn=cost, epsilon=epsilon,
        support=support,
    )
    return prior, mu_0, nu_0, np.asarray(cost), II, JJ


def main(support):
    print(f"Running the full HFPD-OT pipeline on support={support} ...")
    prior, mu_0, nu_0, cost, II, JJ = build_prior(support=support,
                                                  II=10, JJ=10)

    sampler = MetropolisAdjustedLangevinSampler(
        target_log_prob_fn=prior.hyperprior_log_prob_fun,
        target_score_fn=prior.hyperprior_score_fun,
        shape=II * JJ,
        support=support,
        num_parallel_chains=4,
        num_samples=3000,
        num_burnin=1500,
        warm_up_steps=800,
        step_size=0.02,
        seed=0,
        initial_plan=prior.sinkhorn_init(),   # seed chains near the EOT/UOT mode
    )

    print(f"Sampling a {II}x{JJ} = {II * JJ}-dim HFPD-OT plan ...")
    plans, num_accepted, diag, _ = sampler.sample(with_diagnostics=True)
    plans = np.asarray(plans)  # (S, II*JJ), strictly positive
    print(f"  drew {plans.shape[0]} plans of dim {plans.shape[1]}")

    acceptance = np.asarray(num_accepted)[:, -1, 0] / sampler.tot_num_samples
    summary = {
        "dim": II * JJ,
        "num_samples": int(plans.shape[0]),
        "bulk_rhat_max": float(diag["bulk_rhat_max"]),
        "tail_rhat_max": float(diag["tail_rhat_max"]),
        "ess_min": float(diag["ess_min"]),
        "ebfmi_min": float(diag["ebfmi_min"]),
        "acceptance_per_chain": [round(float(a), 3) for a in acceptance],
    }
    print("  diagnostics:", json.dumps(summary, indent=2))

    # Output directory tagged by timestamp.
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "assets" / "images" / f"{run_tag}_{support}"
    outdir.mkdir(parents=True, exist_ok=True)

    # A semantic ordering recovered from the cost geometry (source/target rows of
    # the cost matrix), exercising the dendrogram_order path end-to-end.
    row_order = dendrogram_order(cost, optimal_ordering=True)
    col_order = dendrogram_order(cost.T, optimal_ordering=True)

    plot_transport_plan(plans, II, JJ, mu_0=np.asarray(mu_0), nu_0=np.asarray(nu_0),
                        save_path=str(outdir / "plan_linear.png"))
    plot_transport_plan(plans, II, JJ, mu_0=np.asarray(mu_0), nu_0=np.asarray(nu_0),
                        log_scale=True, save_path=str(outdir / "plan_log.png"))
    plot_transport_plan(plans, II, JJ, mu_0=np.asarray(mu_0), nu_0=np.asarray(nu_0),
                        row_normalize=True, save_path=str(outdir / "plan_rownorm.png"))
    plot_transport_plan(plans, II, JJ, mu_0=np.asarray(mu_0), nu_0=np.asarray(nu_0),
                        row_order=row_order, col_order=col_order,
                        save_path=str(outdir / "plan_reordered.png"))

    (outdir / "diagnostics.json").write_text(json.dumps(summary, indent=2))
    print(f"  artifacts written to {outdir}")
    return outdir


if __name__ == "__main__":
    # Test the full pipeline in both regimes (balanced and unbalanced) and save diagnostics and plots.
    main(support="simplex")
    main(support="positive_orthant")