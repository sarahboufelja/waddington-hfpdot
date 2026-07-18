"""E9 — rank sweep: is mixing driven by the sampling dimension `m` or by the cell count `N`? -> N.

2x2 grid {N in 500,1000} x {r in 2,4}, everything else identical to E8. Since m = r(II+JJ) = 2rN,
the grid gives m in {2000, 4000, 4000, 8000}. BUILT-IN TEST: N=500,r=4 and N=1000,r=2 BOTH have
m=4000 -- if mixing were a function of m alone they must match. They do NOT (0.179 vs 0.070), so
mixing is driven by N (target conditioning), not m; raising r is neutral-to-beneficial.

Reproduce:  python scripts/experiments/e9_rank_sweep.py
Runtime:    ~7 min on CPU (4 devices via XLA_FLAGS).  See docs/experiments.md :: E9.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior

LATENT, EPS = 100, 1e-2
COMMON = dict(num_samples=5000, num_burnin=1000, warm_up_steps=600,
              num_parallel_chains=4, step_size=0.02, seed=0)
MAX_PI_COORDS = 2000
GRID = [(500, 2), (500, 4), (1000, 2), (1000, 4)]   # (cells N, rank r)


def build(N):
    """Synthetic Waddington-shaped target: N cells in a 100-d latent, sq-Euclidean cost."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    src = jax.random.normal(k1, (N, LATENT))
    tgt = jax.random.normal(k2, (N, LATENT)) + 0.5
    C = jnp.sum((src[:, None, :] - tgt[None, :, :]) ** 2, axis=-1)
    C = C / jnp.median(C)
    mu = jnp.ones(N) / N
    nu = 1.3 * jnp.ones(N) / N                     # unbalanced (proliferation)
    return HFPDOTHyperprior(mu_0=mu, nu_0=nu, lambda_1=1.0, lambda_2=1.0, lambda_I_1=0.01,
                            lambda_I_2=0.01, cost_fn=C, epsilon=EPS, support="positive_orthant")


def main():
    for N, r in GRID:
        prior = build(N)
        s = MetropolisAdjustedLangevinSampler(
            target_log_prob_fn=prior.hyperprior_log_prob_fun,
            target_score_fn=prior.hyperprior_score_fun,
            sampling_strategy="low_rank", support="positive_orthant",
            II=N, JJ=N, rank=r, cost=prior.cost_fn, epsilon=EPS, ridge=0.0,
            initial_plan=prior.sinkhorn_init(), **COMMON)
        st = s.sample(max_pi_coords=MAX_PI_COORDS)
        d = st.diagnostics
        eb = np.asarray(d.ebfmi).reshape(-1)
        nacc = np.asarray(st.stacked_mala_states.stacked_num_accepted_samples)[:, -1, 0]
        acc = nacc / s.tot_num_samples
        print(f"N={N:4d} r={r}  params={N * N:>9,}  m=2rN={s.shape:>5}  acc={np.round(np.mean(acc), 2)}  "
              f"eBFMI_MEDIAN={float(np.nanmedian(eb)):.3f}  eBFMI/chain={np.round(eb, 3).tolist()}  "
              f"Rhat_med={float(np.nanmedian(np.asarray(d.bulk_rhat))):.2f}  "
              f"ESS_med={float(np.nanmedian(np.asarray(d.ess))):.0f}")
    print("\nm=4000 twins: N=500,r=4 vs N=1000,r=2 -- if these match, mixing depends on m alone.")


if __name__ == "__main__":
    main()
