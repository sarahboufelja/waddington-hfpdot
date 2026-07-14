# Experiment log — low-rank HFPD-OT sampler: volume term & gauge fixing

Companion to [`mathematical_derivation.md`](mathematical_derivation.md) (the theory). This
file logs what we **ran** and what we **found** — hypotheses, configs, numbers, root causes,
decisions. Newest experiment last. Headline metric is **eBFMI** (energy-space mixing); the
ridge baseline leaves it at ~0.03 (see the main README's Results), and that is what the
volume-term work is trying to move.

> **Convention (mandatory): every entry states its FULL config** — parametrization, mass
> matrix, step size, dims, `num_samples/num_burnin/warm_up_steps`, chains, seed(s), regime(s),
> ε. Comparisons across entries are only valid when a single variable differs; the E2→E3
> "regression" scare came from changing dim **and** run-length **and** step at once. State the
> config, or the number is uninterpretable.

---

## E1 — Option B: pivoted hard-gauge flat section + `½ log det G_S` volume term
*Date: 2026-07-13 · branch: `feature/derive_volume_term`*

### Hypothesis
The v1 ridge is a gauge-*breaking* stopgap that distorts the `π`-marginal and leaves
eBFMI ~0.03. Replacing it with the exact Hausdorff-measure volume term `½ log det G_S` on a
**flat, gauge-fixed section** (Option B, §2.4/§3 of the derivation) should be measure-correct
and *might* improve energy-space mixing. Chosen as the cheap "low-hanging fruit" before the
Option C Stiefel machinery.

### Method
- **Support** `LowRankSection`: pivoted hard gauge — the best-conditioned `r` rows of `U`
  (LU pivot) pinned to `I_r`; free coords `θ = (U_bot, V)`, dim `m = r(II+JJ) − r²`.
  Target `log q = log p(π(θ)) + ½ log det G_S`, `G_S = JᵀJ`, `J = ∂π/∂θ` (exact autodiff,
  `+1e-6 I` jitter). **No ridge** (gauge fixed by the chart).
- **Sampler**: routed through the generic **symmetric-proposal additive MALA** path (the
  support's `name != "low_rank"`), so no bespoke proposal / Jacobian code.
- **Config**: 10×10 orthant HFPD-OT, `r=2`, 4 chains, 800 samples, 200 burn-in, 300 warm-up,
  `step=0.01`, Sinkhorn warm start. Swept `ε ∈ {0.05, 0.2, 0.5}`.

### Correctness (isolated, pre-run) — all pass
- `r=2` Sinkhorn round-trip: `max|π − π_rt| = 2.2e-16`.
- **Subspace motion**: perturbing `U_bot` changes `col(U)` (projector Δ = 0.32) — the property
  the core-only exponential retraction structurally lacked (§3).
- Volume term: autodiff vs finite-diff `|Δ| ≈ 1e-5`.
- `latent_score` vs `jax.grad(latent_log_prob)`: `0.0` (orthant and simplex).

### Sampler bugs fixed to run this (all pre-existing; blocked the generic path for *every* support)
- `MALAState` was not a registered pytree → `lax.scan` rejected the carry. Registered via
  `jax.tree_util.register_dataclass`.
- `rng_key` vs the declared `rng_keys` field mismatch; `step_size` was read from the key field.
- `log_proposal_dist` was called with an unsupported `support_name=` kwarg.

### Results
| ε | acceptance | eBFMI_min | R̂_max | ESS_min |
|---|---|---:|---:|---:|
| 0.05 | 0.57–0.58 | **0.050** | 3.22 | nan |
| 0.20 | 0.56–0.58 | **0.071** | 3.80 | 4 |
| 0.50 | 0.56–0.58 | **0.042** | 3.32 | 4 |

**eBFMI did not move** — ~0.05, statistically indistinguishable from the ridge's ~0.03.
Acceptance is healthy (right on the 0.574 target).

### Root-cause analysis
Healthy acceptance **and** low eBFMI ⇒ well-accepted but *tiny* steps — a random walk under a
stiff, ill-conditioned target. Why stiff:
1. **Hard-gauge asymmetry.** Pinning `U_top = I_r` forces scale into the other factor;
   pivoting relocated the warm-start blow-up `U (‖·‖≈328) → V (‖·‖≈47)`. Directional score
   norms span `128 → 1e10`.
2. **Volume term amplifies it at small ε.** `π = exp(UVᵀ − C/ε)` has huge dynamic range
   (`exp(−20)≈0`), so `J = ∂π/∂θ` rank-collapses → `G` near-singular → `½ log det G` has a
   near-singular Hessian (the jitter stops `nan`s, not the stiffness).
3. **Diagonal mass can't precondition it.** The stiffness is *correlated* (`U_bot ↔ V` couple
   through the bilinear `UVᵀ`), i.e. off-diagonal in the metric; warm-up estimates only
   marginal (diagonal) variances.

### Conclusion
The low eBFMI is at least **partly a preconditioning failure**, confounded by chart stiffness
(as pre-registered) — **not** yet a clean verdict on the volume term. The measure is correct;
the *chart* is a poor sampling geometry.

### Decision — order 3 → 1 → 2
1. **(3)** Full 20×20 bridge run of B for a clean apples-to-apples eBFMI vs the README table
   (removes short-run doubt).
2. **(1)** **Dense mass matrix** (empirical `Cov(θ_warmup)` as the metric) — the systematic
   Stan-style "dense metric" that targets the off-diagonal correlation a diagonal misses. The
   fair test of the volume term. *Confidence: moderate* — a global metric can't absorb the
   position-dependent, 8-orders-of-magnitude curvature from the small-ε `G` singularity; that
   would need the `G(θ)`-metric (Riemannian MALA).
3. **(2)** **Option C** (balanced Stiefel retraction) if eBFMI still doesn't move — the
   symmetric, well-conditioned geometry; removes the asymmetry confound.

### Artifacts
Scratch scripts (not committed): `verify_section.py` (isolated checks), `smoke_section.py`
(E2E smoke), `probe_chain1.py` (freeze/stiffness probe), the ε-sweep one-liner.

---

## E2 — constant dense metric via mode-Hessian whitening (step 1)
*(a fixed preconditioner `M = [−∇²log q(θ*)]⁻¹` with hard-floored-`|λ|` eigenvalues. `M` equals
the Laplace-approximation covariance but is used **only** as a metric — no Gaussian is formed or
sampled, so it is not a Laplace approximation. Also **not** Betancourt's SoftAbs, which is
position-dependent + smooth `λ coth(αλ)`; that's a separate route, and for this manifold the
natural position-dependent metric is `G(θ)=JᵀJ`, not softabs(H).)*
*Date: 2026-07-13 · branch: `feature/derive_volume_term`*

### Hypothesis
E1's low eBFMI is a *preconditioning* failure (correlated `U_bot↔V` stiffness a diagonal mass
can't fix). A dense metric `M = [−∇²log q(θ*)]⁻¹` (SoftAbs at the mode) should whiten the
geometry and move eBFMI. Implemented as `WhitenedSupport` (sample isotropically in `φ`,
`θ = Lφ`, `M = LLᵀ`); zero sampler-internals change.

### Method
10×10 orthant, `r=2`, ε=0.05, 4 chains, **1500 samples, 500 burn-in, 400 warm-up**. Baseline
(diagonal mass, step 0.01) vs Laplace whitening (floor∈{1e-2,1e-3}, step 0.2). Hessian of the
*full* `log q` (incl. volume term) via `jax.hessian` at the Sinkhorn mode.

### Results
| config | acc | eBFMI_min | R̂_max | ESS_min | ESS_med |
|---|---:|---:|---:|---:|---:|
| **diagonal warm-up mass** (the sampler default; NOT unpreconditioned) | 0.58 | **0.213** | 3.12 | 4 | 9 |
| laplace floor=1e-2 | 0.59 | 0.093 | 2.50 | 5 | 12 |
| laplace floor=1e-3 | 0.57 | 0.127 | 2.84 | 5 | 10 |

*(The "baseline" already uses the sampler's diagonal warm-up mass matrix — so this is
**diagonal metric vs dense Laplace-at-mode metric**, and the diagonal one wins. A truly
identity-mass run was not tested.)*

### Findings
1. **eBFMI is config-noisy.** The *same* baseline sampler gave eBFMI 0.05 in E1 (800 samples /
   300 warm-up) and **0.213** here (1500 / 400). So E1's "eBFMI didn't move" was partly a
   short-run artifact — with more draws the volume-term flat-chart sampler already reaches
   eBFMI ~0.2 (vs the ridge's ~0.03). **Small-run eBFMI comparisons are unreliable; the full
   20×20 run (step 3) is required for a stable reference.**
2. **The Laplace mode-metric did *not* help** — eBFMI got *worse* (0.21→0.09–0.13), while
   R̂/ESS_med improved marginally. The whitening *is* working (it let step grow 20× at the same
   acceptance), so this is a real effect, not a bug: bigger accepted steps that don't improve
   *energy* mixing.

### Root cause (E2 finding 2)
A **mode** Hessian is a poor *global* metric when curvature is position-dependent (E1: score
norms span `128→1e10`). It whitens the mode's flat directions and lets the chain take big steps
*along* low-energy-variation directions (helping π-marginal R̂/ESS) while mis-scaling the stiff
directions elsewhere (hurting energy-space eBFMI). This is exactly the regime a **position-
dependent metric `G(θ)`** (Riemannian MALA, experiment 2) is designed for — a constant dense
metric cannot capture it.

### Decision
- A constant dense metric (Laplace-at-mode) is **deprioritized** — it doesn't move eBFMI.
- **Do step 3 first** (full 20×20, baseline diagonal) for a stable eBFMI before any further
  preconditioning — the E1↔E2 baseline swing (0.05↔0.21) shows small runs mislead.
- If preconditioning is revisited, the **`G(θ)`-metric** (position-dependent) is the motivated
  candidate, not a constant one.

---

## E3 — full 20×20 Option B, both regimes, diagonal mass (step 3)
*Date: 2026-07-14 · branch: `feature/derive_volume_term`*

### Method
20×20, `r=2`, section + volume term, diagonal warm-up mass, **README bridge config**
(5000/1000/600, 4 chains, step 0.02), positive-orthant and simplex. The stable, apples-to-apples
comparison against the shipped ridge low-rank baseline.

### Results
| regime | ridge low-rank (v1) | **section + volume (B)** |
|---|---|---|
| unbalanced (orthant) | eBFMI 0.31, ESS 25/899, R̂ 2.43/1.43 | eBFMI **0.067**, ESS nan/10, R̂ 3.56/1.77 |
| balanced (simplex) | eBFMI 0.03, ESS 76/1596, R̂ 1.12/1.04 | eBFMI **0.042**, ESS nan/13, R̂ 2.85/1.67 |

(~355 s orthant, ~388 s simplex.)

> **Simplex caveat:** the section fixes only the `GL(r)` gauge, **not** the simplex's
> softmax-shift gauge (`softmax(ℓ+c𝟙)=softmax(ℓ)`, a redundancy of the softmax layer, and — for
> `r≥2` — a near-flat direction the section leaves open). The section-simplex ran without a
> ridge, so this gauge was unfixed → the 0.042 is **confounded**, not a clean section result.
> The orthant (no shift gauge) is the honest test. See `option_c_proposal.md` §1.1.

### Findings / verdict
- **Option B is a mixing regression vs the ridge**, not an improvement: orthant eBFMI ~5× worse
  (0.067 vs 0.31), ESS collapses to ~10 (from 899) in *both* regimes.
- The E2 10×10 optimism (eBFMI 0.21) **did not survive to 20×20** (0.067) — the method degrades
  with dimension, the wrong direction for the 10⁶ goal.
- Root cause (consistent with E1/E2): the volume term is the correct *measure*, but the
  hard-gauge chart is a stiff, asymmetric *geometry*; constant preconditioning can't fix it.

### Decision
- **Option B (hard-gauge flat chart + volume term) is not the vehicle.** The shipped ridge
  low-rank remains the best sampler.
- This empirically motivates **Option C** (balanced-canonical Stiefel geometry — symmetric,
  well-conditioned) — see [`option_c_proposal.md`](option_c_proposal.md).
- Caveat: B conflates *chart* (hard gauge) with *measure* (volume term). A clean isolation
  (hard-gauge chart **without** the volume term) would separate "chart is stiff" from "volume
  term adds stiffness" — worth one run before fully closing B, if cheap.

> **CORRECTION (see E4).** The "eBFMI regressed 0.213→0.067 with dimension" framing above used
> `eBFMI_min`, which is dominated by the single worst (frozen) chain and is a **noisy summary**.
> The controlled E4 run confirms a *real* dimension effect on the robust **median** (0.20→0.065)
> and via **ESS/R̂**, but the specific min-based numbers should not be compared. Report the eBFMI
> **distribution/median + frozen-chain count**, never `eBFMI_min` alone.

---

## E4 — controlled: vary ONLY dimension (+ seed check), Option B
*Date: 2026-07-14 · branch: `feature/derive_volume_term`*

### Config (full)
Parametrization: `LowRankSection` (pivoted hard-gauge flat chart + volume term `½ log det G_S`,
no ridge), `r=2`. Mass: **diagonal** (warm-up variance). Step: **0.02** (RM-adapted). Draws:
**5000/1000/600**. Chains: **4**. Seeds: **0 and 1**. Regime: **positive orthant only**. ε=0.05.
Dims: **10×10 and 20×20** (the only variable that changes vs seed).

### Results (per-chain eBFMI, to expose heterogeneity)
| run | per-chain eBFMI | median | eBFMI_min | R̂_max | ESS_med |
|---|---|---:|---:|---:|---:|
| 10×10 s0 | [0.24, 0.20, 0.09, 0.07] | 0.14 | 0.067 | 3.17 | 10 |
| 10×10 s1 | [0.21, 0.32, 0.30, **0.00**] | 0.26 | 0.000 | 2.79 | 7 |
| 20×20 s0 | [0.17, 0.08, **0.00**, 0.04] | 0.06 | 0.000 | 3.52 | 8 |
| 20×20 s1 | [0.05, 0.22, 0.03, 0.09] | 0.07 | 0.033 | 3.39 | 10 |

### Findings
1. **`eBFMI_min` is a poor summary** — dominated by the worst chain; some chains **freeze**
   (eBFMI→0) intermittently and seed-dependently. This is what made E1/E2/E3's min-values swing.
2. **Real dimension effect on the median**: 10×10 median eBFMI ≈ 0.20 (both seeds) → 20×20 ≈
   0.065 (both seeds); even the best chain drops (0.32→0.22). ~3× degradation, consistent across
   seeds (n=2, indicative not definitive). So B *does* degrade with dimension.
3. **Robust gap vs the ridge**: ESS_med ~8–10, R̂ ~3 at both dims (ridge: ESS 899, R̂ 1.43). This
   ESS/R̂ signal — not `eBFMI_min` — is the solid "B mixes worse" evidence.

### Root cause / decision
Intermittent chain freezing = the hard-gauge stiffness biting individual chains; the median
degradation with dimension = the same stiffness worsening as `m` grows. Both point to the same
fix: **Option C's well-conditioned symmetric geometry**. Methodology going forward: **eBFMI
median + per-chain spread + frozen-chain count**, not `eBFMI_min`.
