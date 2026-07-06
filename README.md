# waddington-hfpdot

**A scalable, diagnostics-first MCMC sampler for high-dimensional randomized optimal-transport plans.**

`waddington-hfpdot` samples the **HFPD-OT hyperprior** — the optimal Bayesian
distribution over transport plans from *Randomized transport plans via hierarchical
fully probabilistic design* (Boufelja Y., Quinn & Shorten, *Information Sciences*
718, 2025) — and its **unbalanced (Waddington) extension** for mass-varying processes
such as cell proliferation and apoptosis.

Where classical optimal transport returns a single, certainty-equivalent plan, HFPD-OT
returns a *distribution* over plans, turning transport design into Bayesian inference question:
you get an expected plan **and** calibrated uncertainty on every contract `π_ij`. This
library makes drawing from that distribution practical — and is being built toward the
`~10⁶`-dimensional plans of single-cell lineage inference (the Waddington landscape).

---

## Why it's hard, and what's here

The HFPD-OT hyperprior is a **smooth, unimodal, but severely ill-conditioned** density
on the probability simplex (balanced) or positive orthant (unbalanced). Vanilla
Metropolis-Adjusted Langevin (MALA) sampling of the full `II·JJ` plan entries degenerates and mixing collapses (ESS in single digits).

This library addresses that with three ideas, each a self-contained, tested module:

- **Constrained-support reparametrizations** (`supports.py`) — a clean `Support`
  abstraction that owns the change of variables (`to_constrained`/`to_unconstrained`,
  log-Jacobian, and the lifted `latent_log_prob`/`latent_score`). The sampler is
  support-agnostic; the geometry lives entirely in the `Support`.
- **Low-rank reparametrization** (`LowRank`) — instead of sampling the full
  `II·JJ` plan, sample low-rank factors `θ=(U,V)` with `log π = U Vᵀ − C/ε`. This
  generalizes the entropic-OT dual Kantorovitch potentials (`r=2` reproduces the Sinkhorn plan),
  collapses the sampled dimension to `r(II+JJ)`, and — the real additivity — **reconditions
  the geometry** so the drift can dominate the noise.
- **Arviz-grade diagnostics** (`mcmc_diagnostics.py`) — rank-normalized split-R̂
  (Vehtari et al. 2021), FFT/Geyer ESS (Stan recipe), and eBFMI, all validated against
  ArviZ, computed per-chain on the gauge-invariant plan `π`.

---

## Results

On a 20×20 problem (`ε=0.05`, 4 chains × 5000 draws), assessed on the plan `π`:

| regime | sampler | eBFMI | R̂ (max/med) | ESS (min/med) |
|---|---|---:|---:|---:|
| unbalanced (orthant) | vanilla MALA, 400-d | 0.09 | 1.37 / 1.02 | 19 / 243 |
| unbalanced (orthant) | **low-rank r=2, 80-d** | **0.31** | 2.43 / 1.43 | 25 / **899** |
| balanced (simplex) | vanilla MALA, 400-d | 0.00 | 3.11 / 1.29 | **5 / 17** |
| balanced (simplex) | **low-rank r=2 + gauge ridge** | 0.03 | **1.12 / 1.04** | 76 / **1596** |

The low-rank reparametrization delivers **3–4× the effective samples** on the
unbalanced regime, and on the hardest (balanced/simplex) case rescues a **broken** vanilla
sampler (ESS 5–17, R̂ 3.1) into one whose **plan marginals mix well** (ESS ~1600, R̂ 1.04).
Critically, we **do not claim convergence** at this stage: the low **eBFMI (0.03)**, well under the ~0.3
healthy threshold, shows energy-space exploration is still poor. R̂ and ESS are *marginal*
diagnostics on `π`; eBFMI is the energy signal, and it is the honest one here — improving it
is open work (see Roadmap). See [`docs/mathematical_derivation.md`](docs/mathematical_derivation.md)
for the mathematics.

Reproduce: `python scripts/run_lowrank_bridge.py`.

---

## Quickstart

```bash
uv sync --all-groups --all-packages --all-extras       # Install dependencies.
```

```python
import jax.numpy as jnp
from langevin_sampler import HFPDOTHyperprior, MetropolisAdjustedLangevinSampler

II = JJ = 20
# A structured cost + bell-shaped nominal marginals (source/target on [0,1]).
cost = ...                               # (II, JJ), normalized to [0, 1]
mu_0, nu_0 = ..., ...                    # probability vectors (balanced) or masses (unbalanced)

prior = HFPDOTHyperprior(
    mu_0=mu_0, nu_0=nu_0,
    lambda_1=1.0, lambda_2=1.0, lambda_I_1=0.01, lambda_I_2=0.01,
    cost_fn=cost, epsilon=0.05,
    support="positive_orthant",          # or "simplex" (balanced)
)

# Low-rank sampler (Option B): sample r(II+JJ) factors, not II*JJ entries.
sampler = MetropolisAdjustedLangevinSampler(
    target_log_prob_fn=prior.hyperprior_log_prob_fun,
    target_score_fn=prior.hyperprior_score_fun,
    sampling_strategy="low_rank", support="positive_orthant",
    shape=2 * (II + JJ), II=II, JJ=JJ, rank=2, cost=prior.cost_fn, epsilon=0.05,
    initial_plan=prior.sinkhorn_init(),  # warm-start near the EOT/UOT mode
    num_samples=5000, num_burnin=1000, warm_up_steps=600,
    num_parallel_chains=4, step_size=0.02, seed=0,
    ridge=0.1,                           # gauge-breaking ridge (needed for the simplex)
)

state = sampler.sample()                 # FinalMalaState
plans = state.samples                    # posterior draws of the plan pi
print(state.diagnostics.bulk_rhat_max, state.diagnostics.ess_min, state.diagnostics.ebfmi_min)
```

---

## Architecture

| module | responsibility |
|---|---|
| `langevin_sampler.py` | MALA kernel: preconditioned Langevin + MH, warm-up mass matrix, Robbins-Monro step-size adaptation, multi-chain (`vmap`/sharding). `HFPDOTHyperprior` (log-prob, analytic score, Sinkhorn warm start). |
| `supports.py` | `Support` change-of-variables: `Unconstrained`, `PositiveOrthant`, `Simplex`, `LowRank`. Analytic scores tested against `jax.grad`. |
| `mcmc_diagnostics.py` | `MCMCDiagnostics` / `DiagnosticsSummary`: split-R̂, ESS, eBFMI (per-chain, gauge-invariant on `π`). |
| `transport_summary.py`, `sampler_viz.py` | Plan summaries (mean/std, credible bands, dendrogram ordering) and matplotlib-free/lazy plotting. |
| `scripts/run_lowrank_bridge.py` | The head-to-head experiment (vanilla vs low-rank, both regimes). |

Design principles: **separation of concerns** (the sampler never sees support internals),
**symmetry** (every reparametrization implements the same `Support` interface, and
tensor ops are batch-aware) and **diagnostics-first** (convergence is measured, on the
right space, not assumed).

---

## Software & hardware

- **Software.** Built on **JAX** — `jit` + `vmap` + `jax.sharding` (a `Mesh` over the
  available devices) drive the multi-chain MALA kernel and the batch-aware `Support`
  transforms from a single code path, on CPU or GPU. Python ≥ 3.10, dependencies pinned
  with **uv** (`pyproject.toml` + `uv.lock`).
- **Hardware.** Developed and benchmarked on an **HP Z8 Fury G5 Workstation** with **two
  NVIDIA GPUs**; chains shard one-per-device (the 4-chain runs above map directly onto the
  two-GPU box), with a clean fallback to CPU.

---

## The gauge, and the ridge

The low-rank factorization has a `GL(r)` gauge (`(U,V) → (UR, VR⁻ᵀ)` leaves `π`
unchanged). For the **simplex** there is an additional, **unbounded** gauge — softmax
shift-invariance, `softmax(ℓ + c𝟙) = softmax(ℓ)` — which lets `‖θ‖` run to infinity while
`π` stays frozen. The current release breaks it with a **gauge ridge** `−λ‖θ‖²`
(`ridge=`), which pins the runaway and restores healthy R̂/ESS on `π` — though eBFMI stays
low, so this is *marginal* mixing, **not** convergence (see Results). This is a deliberate v1
stopgap: it is exact enough to sample well but distorts the `π`-marginal slightly. The
principled replacements are on the roadmap.

---

## Roadmap

- **Volume term** `½ log det G` — the correct measure on the low-rank manifold;
  handles the `GL(r)` and softmax-shift gauges uniformly (replaces the ridge), via
  matrix-free stochastic log-determinant estimation for scaling.
- **Pluggable kernel** — HMC / MCLMC on the reconditioned low-rank coordinates for the
  `~10⁶`-dim Waddington runs.
- **Exact condition-number derivation** (primal vs. dual) to formalize the reconditioning claim.
- **DVC** for the single-cell genomics dataset (kept out of the code repo).

---

## References

- S. Boufelja Y., A. Quinn, R. Shorten. *Randomized transport plans via hierarchical
  fully probabilistic design*. **Information Sciences** 718 (2025) 122365.
- A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, P.-C. Bürkner. *Rank-normalization,
  folding, and localization: an improved R̂...*. **Bayesian Analysis** 16(2), 2021.
- M. Cuturi. *Sinkhorn distances*. **NeurIPS** 2013. · G. Peyré, M. Cuturi.
  *Computational Optimal Transport*. **FnT ML** 2019.

## License and citation

Released under the **MIT License** ([`LICENSE`](LICENSE)) — © 2026 Sarah Boufelja Yacobi and
Imperial College London.

If you use this software, please cite **both** the software and the HFPD-OT paper above. A
machine-readable [`CITATION.cff`](CITATION.cff) is included (GitHub renders a "Cite this
repository" button from it).
