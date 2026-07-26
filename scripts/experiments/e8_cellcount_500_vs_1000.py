"""E8 (Part B) — proper multi-chain cell-scale re-validation: 500/day (250k) vs 1000/day (10^6).

Identical config, only the cell count N differs, at rank r=2. Full README length (5000/1000/600),
4 chains, in-repo sampler, orthant (unbalanced), ridge=0, synthetic Waddington-shaped target
(N cells -> 100-d latent Gaussian clouds -> sq-Euclidean cost, median-normalised). Per-parameter-
thinned diagnostics (max_pi_coords=2000, E7); eBFMI from carried energies (E8 energy carry).

Reproduce:  python scripts/experiments/e8_cellcount_500_vs_1000.py
Runtime:    ~4 min on CPU (4 devices via XLA_FLAGS).  See docs/experiments.md :: E8.
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
from device_info import GpuMonitor, print_device_banner
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior

LATENT, EPS = 100, 1e-2
COMMON = dict(num_samples=5000, num_burnin=1000, warm_up_steps=600,
              num_parallel_chains=4, step_size=0.02, seed=0)
MAX_PI_COORDS = 2000


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
    # Announce the hardware: a silent CPU fallback turns every timing below into a CPU timing.
    print_device_banner()
    with GpuMonitor() as monitor:
        _run()
    print(monitor.summary(), flush=True)


def _run():
    for N in (500, 1000):
        prior = build(N)
        s = MetropolisAdjustedLangevinSampler(
            target_log_prob_fn=prior.hyperprior_log_prob_fun,
            target_score_fn=prior.hyperprior_score_fun,
            sampling_strategy="low_rank", support="positive_orthant",
            II=N, JJ=N, rank=2, cost=prior.cost_fn, epsilon=EPS, ridge=0.0,
            initial_plan=prior.sinkhorn_init(), **COMMON)
        st = s.sample(max_pi_coords=MAX_PI_COORDS)
        d = st.diagnostics
        eb = np.asarray(d.ebfmi).reshape(-1)
        nacc = np.asarray(st.stacked_mala_states.stacked_num_accepted_samples)[:, -1, 0]
        acc = nacc / s.tot_num_samples
        rhat, ess = np.asarray(d.bulk_rhat), np.asarray(d.ess)
        print(f"N={N:4d}  params={N * N:>9,}  m=2rN={s.shape:>5}  acc={np.round(np.mean(acc), 2)}  "
              f"eBFMI_MEDIAN={float(np.nanmedian(eb)):.3f}  eBFMI/chain={np.round(eb, 3).tolist()}  "
              f"Rhat_med={float(np.nanmedian(rhat)):.3f}  ESS_med={float(np.nanmedian(ess)):.0f}")
        # NB: quote MEDIANS only; Rhat_max/ESS_min are optimistically biased under trimming (see E7).


if __name__ == "__main__":
    main()
