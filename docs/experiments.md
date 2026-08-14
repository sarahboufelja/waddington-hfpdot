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
>
> **Convention (mandatory): report the MEDIAN, never `eBFMI_min` / `R̂_max` / `ESS_min` alone.**
> Two independent reasons, both learned the hard way:
> 1. **Noise (E4)** — `eBFMI_min` is the single worst chain, often a frozen one; it swung
>    0.05/0.213/0.067 across runs of the *same* sampler.
> 2. **Bias (E7)** — under **per-parameter thinning** the extremes are *systematically
>    optimistic*: `R̂_max` over a coordinate subset can only be **≤** the true max, and `ESS_min`
>    can only be **≥** the true min. At 20×20 with 100/400 coords the effect is small (2.95 vs
>    2.99); at 10⁶ with 2000/1,000,000 coords (**0.2%**) we will essentially never sample the worst
>    coordinate, so `R̂_max`/`ESS_min` are **not worst-cases at all**. The median is unbiased.
>
> eBFMI is exempt from (2): it reads only `theta`, so trimming cannot touch it (verified
> bit-identical, E7).
>
> **⚠ CRITICAL (2026-07-19): E5–E9 ALL RAN ON CPU.** Every experiment set `JAX_PLATFORMS=cpu` with
> 4 *simulated* host devices, on a machine with **2× RTX 4070 Ti SUPER that were never used**.
> - **Still valid:** eBFMI / R̂ / ESS are properties of the Markov chain (target, proposal, step size,
>   seed) — hardware-independent. E8's 500-vs-1000 and E9's "N not m" conclusions stand.
> - **NOT valid as stated:** every timing, and the whole feasibility/compute envelope. In particular
>   "neither cell count converged at 5000 draws" may be an artifact of a CPU-sized budget — R̂ was
>   still falling (1.91→1.79→1.73→1.66) and ESS still climbing (20→371) when we stopped.
> - **Consequence:** **500 cells/day is a LOWER BOUND on the budget, not the answer.** Revisit after
>   the pipeline lands, via a GPU calibration + long-run convergence test.
>
> **Convention (mandatory, from E10 onward): every script prints a device banner** (`device_info.
> print_device_banner()`) and wraps runs in `GpuMonitor`, so a silent CPU fallback — or a run that
> works only one of two GPUs — is visible in the output instead of being discovered weeks later.
>
> **Convention (mandatory, from E8 onward): every entry includes a `Reproduce:` line** — the exact
> CLI command for a **committed** script under `scripts/experiments/` (the scratchpad is ephemeral).
> One command, from the repo root, no reconstruction. (Earlier entries E1–E7 predate this; back-fill
> on next touch.)
>
> **Fixed across the ENTIRE log unless stated: rank `r = 2`.** The sampling dimension is
> `m = r(II+JJ) = 2rN`, BUT — proven by the rank sweep (**E9**) — **`m` is NOT the mixing lever**:
> the `m=4000` configs `N=500,r=4` (eBFMI 0.179) and `N=1000,r=2` (0.070) differ 2.5×. Mixing is
> driven by **`N` (cell count / plan size), a target-conditioning effect**, and raising `r` is
> **neutral-to-beneficial**. So **rank and cell-count are DECOUPLED for mixing** — higher rank (for
> expressivity, [[the low-rank representational-power note]]) carries no mixing penalty. Every
> number here is still at `r=2`; the required rank on the real target is open but not a mixing risk.

---

## E9 — rank sweep: is mixing driven by `m` (sampling dim) or by `N` (cell count)? → `N`, NOT `m`
*Date: 2026-07-16 · branch: `feature/derive_volume_term`* — **corrects the `m`-curse mechanism asserted in E8.**

### Design
`m = r(II+JJ) = 2rN`. 2×2 grid {N∈500,1000} × {r∈2,4}, everything else = E8 (orthant, ridge=0,
synthetic Waddington-shaped target, in-repo sampler, 4 chains, 5000/1000/600, step 0.02, seed 0,
`max_pi_coords=2000`). **Built-in test:** `N=500,r=4` and `N=1000,r=2` both have **m=4000** — if
mixing is a function of `m` alone they must match.

**Reproduce:** `python scripts/experiments/e9_rank_sweep.py` (~7 min, CPU).

### Results
| N | r | params | **m=2rN** | acc | **eBFMI med** | eBFMI/chain | ESS med |
|---|---|---|---|---|---:|---|---:|
| 500 | 2 | 250k | 2000 | 0.59 | **0.172** | [0.17,0.14,0.17,0.18] | 559 |
| 500 | 4 | 250k | **4000** | 0.60 | **0.179** | [0.19,0.17,0.19,0.08] | 460 |
| 1000 | 2 | 1M | **4000** | 0.59 | **0.070** | [0.07,0.05,0.07,0.09] | 371 |
| 1000 | 4 | 1M | 8000 | 0.58 | **0.122** | [0.10,0.10,0.15,0.14] | 341 |

### Findings — `m` is NOT the mixing lever
- **The m=4000 twins differ 2.5×** (0.179 @N=500,r=4 vs 0.070 @N=1000,r=2). Same sampling dimension,
  very different mixing → **eBFMI is not a function of `m`.** *(This falsifies E8's "MALA
  `step∝dim^{-1/3}` curse on `m`" — kept there with a correction pointer.)*
- **Raising `r` is neutral-to-beneficial, never harmful**: N=500 `0.172→0.179` (flat despite 2× `m`);
  N=1000 `0.070→0.122` (**better** despite 2× `m`). So the naive MALA dimension curse does not bite
  here (step adaptation holds acceptance ~0.58; the extra low-rank directions are soft, not stiff).
- **The driver is `N` (cell count / plan size `n=N²`)**: ~0.17 @N=500 vs ~0.07–0.12 @N=1000,
  regardless of `r`. Not a sampler-dimension effect → a **target** effect: the HFPD-OT energy
  landscape sharpens / ill-conditions as the plan grows at fixed ε=1e-2. *(Mechanism = hypothesis;
  ε is the obvious co-lever to probe.)*

### Implications
- **Rank and cell-count are DECOUPLED for mixing.** Higher `r` (needed for plan expressivity —
  [[the low-rank representational-power note]]) carries **no mixing penalty**, even helps at high N.
  Removes the "rank hurts mixing" worry.
- Mixing lever is **`N` via target conditioning**, so 500 cells still mixes better than 1000 (E8) —
  but the reason is conditioning, not `m`.
- **Higher `r` partially rescues mixing at high cell counts** (N=1000: r=4 → 0.122 vs r=2 → 0.070).
- OPEN: confirm the ε / target-conditioning mechanism; sweep on the real GMVAE-derived target.

---

## E8 — energy-carry refactor (80 GB leak fixed) + PROPER cell-scale re-validation (500 vs 1000)
*Date: 2026-07-16 · branch: `feature/derive_volume_term`* — **supersedes E5's preliminary at-scale numbers.**

### Part A — the eBFMI 80 GB leak, and the fix (energy carry)
E5 never called `sample()` (hand-rolled loop). The in-repo `sample()` path had a *second* copy of
the E7 wall, in the eBFMI computation: `MCMCDiagnostics.ebfmi` recomputed energies via
`jax.vmap(jax.vmap(log_prob))(chains)`, and each `log_prob` internally does `to_constrained(theta)`
→ a full plan. Vectorised over all `(C, D)` draws, the intermediate is `(C, D, II*JJ)` = **80 GB at
10⁶** → OOM (killed ~1000 draws; RSS grew ~58 MB/draw). It never surfaced at 20×20 (32 MB).
**Fix (`MALAState.latent_log_prob_state`):** the accepted state's log-prob is **already computed for
the MH ratio** every step, so carry it through the `lax.scan` and hand the stacked `(C, D)` energies
straight to `ebfmi` — **zero recompute, zero plan materialisation** at diag time. (A *sequential*
recompute would also fit memory; the carry additionally saves the compute and is exact for the
states visited.) `ebfmi` now consumes energies, not `log_prob_fn`. **Result: memory flat in draws —
5000 draws @ 10⁶ in 132 s at 15.6 GB** (was un-runnable). 56 tests green.

### Part B — proper multi-chain cell-scale re-validation
**Config (full):** synthetic Waddington-shaped target (N cells → 100-d latent Gaussian clouds →
sq-Euclidean cost, median-normalised; unbalanced uniform marginals `nu=1.3/N`), ε=1e-2. `LowRank`
**orthant**, r=2, **ridge=0**. **In-repo sampler** (warm-up diagonal mass matrix + Robbins–Monro),
**4 chains, 5000 samples / 1000 burn-in / 600 warm-up**, step 0.02, seed 0. Per-parameter-thinned
diagnostics `max_pi_coords=2000` (E7). eBFMI from carried energies.

**Reproduce (Part B):** `python scripts/experiments/e8_cellcount_500_vs_1000.py` (~4 min, CPU).

| N (cells) | params | m=r(II+JJ) | acc | **eBFMI median** | eBFMI/chain | R̂ med | ESS med | time |
|---|---|---|---|---:|---|---:|---:|---|
| 500 | **250k** | 2000 | 0.58–0.61 | **0.172** | [0.17,0.14,0.17,0.18] | 1.64 | 559 | 81 s |
| 1000 | **10⁶** | 4000 | 0.58–0.60 | **0.070** | [0.07,0.06,0.07,0.09] | 1.66 | 371 | 139 s |

*(Both: 0/4 frozen chains. Biased-optimistic extremes, do not quote: 500 → R̂_max 2.61/ESS_min 25;
1000 → R̂_max 2.67/ESS_min 13.)*

### Findings
- **Halving cells ~2.5×'s eBFMI** (0.070→0.172) at equal acceptance. Ladder: **20×20 → 0.31,
  250k → 0.172, 10⁶ → 0.070**, all at **r=2**.
  > **CORRECTION (E9):** I first attributed this to a MALA `step∝dim^{-1/3}` curse on `m=2rN`. **The
  > rank sweep (E9) falsifies that** — the `m=4000` configs `N=500,r=4` and `N=1000,r=2` differ 2.5×,
  > and raising `r` (i.e. `m`) does NOT hurt. The driver is **`N` (cell count / plan size), a target-
  > conditioning effect**, not the sampling dimension `m`. Rank and cell-count are **decoupled** for
  > mixing. So the 500>1000 advantage is real but is about *conditioning*, not dimension.
- **Neither is converged** at 5000 draws (R̂ median ≈ 1.64 — the *unbiased* median, want <1.01). So
  the choice is "meaningfully better-mixed (500)" vs "poorly-mixed (1000)", not converged vs not.
- **E5 was optimistic** exactly as flagged: its single-chain 0.268 @10⁶ vs this 4-chain 0.070.

### Decision
- **Primary = 500 cells/day (250k).** 2.5× the eBFMI, higher ESS, comfortable memory.
- **1000 cells/day = documented stretch** — runs now (memory fixed) but mixes poorly; trustworthy UQ
  there needs longer chains or the deferred mixing work.
- **Revisit after the Waddington E2E pipeline exists**: the pipeline's scientific conclusions decide
  whether **Option C** (Stiefel) goes into *this* paper or defers to a separate methods paper.

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

## E7 — per-parameter thinning: unblocking diagnostics at 10⁶ (the 80 GB wall)
*Date: 2026-07-15 · branch: `feature/derive_volume_term`*

### The wall
`sample()` expanded **every** stored `theta` into its whole plan at once for the R̂/ESS diagnostics:
`chains × draws × II*JJ` = **80 GB** at 10⁶ (4 chains × 5000 draws), vs **56 GB available**. The
chain itself is compact (`theta` ≈ 320 MB — 250× smaller than the plans); we were throwing away
exactly the compression low-rank buys us, at the last step, purely for bookkeeping. It never
surfaced before: the same line costs **32 MB** at 20×20, and E5 (our only at-scale run) used a
hand-rolled loop that stored only `theta` and **never called `sample()`**.

### Fix: `sample(max_pi_coords=k)` — per-parameter thinning
Keep a **fixed** random `k`-coordinate subset of `pi`, drawn **once** and closed over, so the *same*
coordinates are kept in every draw. That fixedness is what makes it sound: each retained coordinate
keeps its **complete draw-trace**, so its autocorrelation — hence `ESS_j`/`R̂_j` — is **exact and
uncapped**. (A fresh subset per draw would give ragged traces and meaningless ESS.) Plans are
rebuilt **one at a time** via `lax.map` over the flattened `(C*D, m)`, so only one ~4 MB plan is
ever live. `latent_samples` (the `theta` archive) is now always returned.

**Why this axis, not draw-thinning?** Both are valid; they are the *two regimes*:
| regime | reduction | ESS obtained |
|---|---|---|
| diagnostics (here) | fixed coord subset × **all draws** | **exact, uncapped** |
| downstream lineage | **thinned draws** × whole plans | capped at retained draw count |
Draw-thinning (Patterson & Teh, NIPS 2013, thin=100 → ESS ≤ 1000) *also* fixes the memory and keeps
whole plans, but caps ESS. Per-parameter thinning gives the uncapped ESS the mixing question needs.
A per-parameter-thinned `samples` is **diagnostics-only** — it cannot be push-forwarded.

### Validation
| config | samples | eBFMI med | R̂ max | ESS med |
|---|---|---:|---:|---:|
| 20×20 untrimmed | (4, 800, **400**) | 0.555 | 2.99 | 89 |
| 20×20 trimmed(100) | (4, 800, **100**) | **0.555** (bit-identical) | 2.95 | 81 |
| **1000×1000 trimmed(2000)** | (4, 400, 2000) | 0.136 | 4.57 | 23 |

- **eBFMI bit-identical** — it reads only `theta`, so trimming structurally cannot touch it.
- **10⁶ RAN in 24 s** where 80 GB was impossible. Blocker gone; 57 tests still green (`None` default
  preserves prior behaviour).

### Finding: the extremes are optimistically BIASED under trimming
`R̂_max` 2.95 < 2.99 and `ESS_med` 81 < 89 are not noise — they're systematic. A max over a subset
can only be **≤** the true max; a min can only be **≥** the true min. At 20×20 (100/400 = 25%) the
effect is small; **at 10⁶ (2000/1,000,000 = 0.2%) we will essentially never sample the worst
coordinate**, so `R̂_max`/`ESS_min` are *not worst-cases*. Only the **median** is unbiased — a second,
independent reason for the median rule (E4's was noise). See the convention block at the top.

### Note
The 10⁶ row is a **short (400-draw) smoke**, not the re-validation: eBFMI median **0.136** is about
half E5's single-chain 0.268 (consistent with E5 being optimistic), but 400 draws cannot support
R̂/ESS. The proper 1M re-validation (multi-chain, both regimes, realistic draw count) is still owed.

---

## E6 — D1 GATE: does the re-routed IN-REPO sampler reproduce the README 20×20? ✅ PASS
*Date: 2026-07-15 · branch: `feature/derive_volume_term`*

### Why
D1 re-routed the `low_rank` path off the dead balanced-gauge retraction onto the generic **additive
MALA** path (+ removed a leftover in `log_proposal_dist`, + fixed the `with_ridge` contract bug).
Gate: the *in-repo* sampler must reproduce the shipped README numbers — no regression.

### Config (full)
Real HFPD-OT **20×20** target (structured quadratic cost normalized to [0,1], bell marginals,
ε=0.05), `sampling_strategy="low_rank"`, r=2 (m=80), **ridge 0.1 simplex / 0 orthant**, Sinkhorn
warm start, **4 chains**, 5000/1000/600, step 0.02, seed 0, diagonal warm-up mass. In-repo sampler.

### Results (E4 methodology: median + per-chain + frozen count, NOT `eBFMI_min`)
| metric | README orthant | **got** | README simplex | **got** |
|---|---|---|---|---|
| eBFMI | 0.31 *(min)* | **med 0.327**, min 0.173, per-chain [0.277, 0.173, 0.480, 0.376] | 0.03 *(min)* | **med 0.033**, min 0.028, per-chain [0.028, 0.028, 0.038, 0.037] |
| R̂ max/med | 2.43 / 1.43 | **2.27 / 1.42** | 1.12 / 1.04 | **1.10 / 1.02** |
| ESS min/med | 25 / 899 | **17 / 935** | 76 / 1596 | **31 / 1392** |
| frozen (<0.02) | — | **0/4** | — | **0/4** |
| acc/chain | — | 0.57–0.60 | — | 0.57–0.59 | 

(~10 s orthant, ~5 s simplex.)

### Findings
- **PASS — no regression.** R̂ and ESS-median land essentially on the README; the re-route to
  additive MALA is correct.
- **The simplex 0.03 is REAL and ROBUST — the `with_ridge` fix did NOT move it** (0.033 vs 0.03).
  So the missing ridge-in-the-drift was *not* the cause of the simplex's energy-mixing failure;
  it is something more fundamental (softmax-shift gauge / the ridge's own distortion). A plausible
  lever, now ruled out.
- **Orthant/simplex inversion**: orthant eBFMI 0.33 but R̂ 2.27; simplex R̂ 1.10 / ESS 1392 but
  eBFMI 0.033. Confirms the standing rule: R̂/ESS are **marginal** diagnostics on `π`; eBFMI is the
  **energy** signal — never claim convergence from R̂/ESS alone.
- E4's lesson recurs: our orthant *min* (0.173) vs README's *min* (0.31), with per-chain spread
  0.17–0.48 → the **median** is the stable comparator.
- **The Waddington delivery is orthant** (unbalanced: proliferation/apoptosis) = our **strong**
  regime (median eBFMI 0.33, 0 frozen chains).

### Decision
D1 complete. Proceed to D2–3 (Waddington layer review). The at-scale (1M) setting is **still
provisional** — see E5's TODO: revisit multi-chain, both regimes, in-repo sampler, real target.

---

## E5 — ridge low-rank sampler SCALING to cell-level (250k / 10⁶ params)
*Date: 2026-07-14 · branch: `feature/derive_volume_term`*

> **SUPERSEDED by E8.** E5 is a **single-chain, hand-rolled** probe; its at-scale eBFMI (0.268 @10⁶)
> is optimistic. The proper 4-chain in-repo numbers are in E8 (0.070 @10⁶, 0.172 @250k). Keep E5
> only for the compute/memory scaling and the warm-start NaN root-cause; use **E8** for mixing.

### Why
The Waddington delivery is **cell-level** (unit = cells; population tables are aggregations of
cell-level lineage; both tables + FLE required). Scale = `n_cells²`: 250k (500 cells/day) or
10⁶ (1000). Decision question: does the **ridge** low-rank sampler (analytic `O(n·r)` score, NO
volume term) scale? — because Option B/C's volume term is `O(m³)`, un-runnable at `m~4000`.

### Config (full)
Synthetic Waddington-shaped target (100-d latent clouds → sq-Euclidean cost, median-normalized,
unbalanced uniform marginals), ε=0.01. `LowRank` orthant, r=2, ridge=0. **Additive MALA** with
Robbins–Monro step adaptation (target acc 0.574), 1 chain, 2000 steps, hand-rolled (the sampler's
`low_rank` branch still routes through the dead retraction code). eBFMI on 2nd-half energies. CPU.

### Results
| N | params | m=r(II+JJ) | score/step | plan mem | acc | eBFMI | 2000 steps |
|---|---|---|---|---|---:|---:|---|
| 200 | 40k | 800 | 0.2 ms | 0.2 MB | 0.58 | **0.653** | 2 s |
| 500 | **250k** | 2000 | 0.8 ms | 1.0 MB | 0.59 | **0.272** | 7 s |
| 1000 | **10⁶** | 4000 | 2.4 ms | 4.0 MB | 0.58 | **0.268** | 26 s |

*(First pass had N=1000 = NaN. Root-caused to a real bug — **not** `exp` overflow — and fixed; the
table above is post-fix.)*

### The N=1000 NaN: a guard that didn't guard (real bug, now fixed)
Hypothesis was `exp` overflow. **Wrong** — `ell = UVᵀ−C/ε` maxes at **−7.1**, nowhere near float32's
+88 overflow. Actual chain:
1. Sinkhorn at N=1000/ε=0.01 **underflows to exactly 0** in 2 entries (at N=500 the min was
   `6.99e-38`, just above float32's smallest normal `1.18e-38`, so it survived — a classic
   "works at the size it was tested at" bug).
2. `to_unconstrained` guarded with `jnp.clip(pi, 1e-300, None)` — but **`1e-300` is below float32's
   smallest subnormal, so the floor itself underflows to `0.0` and the clip is a NO-OP.**
3. `log(0) = -inf` → SVD receives `-inf` → LAPACK `SLASCL` error → `theta0 = NaN` → all NaN from
   step 0 (hence `|dθ|=nan`, not a frozen chain).

**Fix:** dtype-aware floor `jnp.finfo(pi.dtype).tiny` (→ `log ≈ -87`, finite) in **both**
`LowRank.to_unconstrained` and `LowRankSection.to_unconstrained`. This changes **nothing** about the
sampled target — it only stops `-inf` reaching the SVD. *Lesson: sweep for other float64-scale
constants used in float32 paths.*

### Findings
- **Compute + memory scale linearly** (`O(n·r)`): 2.4 ms/step at 10⁶ on CPU, 4 MB/plan. Not a
  blocker (**stream** plans; never materialize thousands — 4 MB × 5000 = 20 GB). Faster on GPU.
- **Mixing HOLDS at cell scale**: eBFMI **0.272 @250k → 0.268 @10⁶** — it *plateaus*, it does not
  keep degrading (the drop is 40k→250k, then flat); acceptance on target (0.58) at both.
- **Comparability — read this before quoting any number.** E5 is **orthant** (ridge=0). Its correct
  comparator is the README's **orthant** low-rank = **0.31**, so E5 ≈ *reproduces* that at 600–2500×
  the size. It is **NOT** an improvement on the README's **0.03**, which was the **simplex** (a
  different regime, ridge=0.1) — **the simplex has never been re-measured at scale.** Nor is E5
  cleanly comparable to Option B's 0.067 (E3): different target (synthetic vs real), statistic
  (1-chain vs 4-chain `eBFMI_min`), and measure (no volume term vs volume term).
- **The genuine E5 finding** is therefore: *orthant mixing does not degrade with scale*
  (0.31 @400 params → 0.27 @250k → 0.27 @10⁶) — "holds up", not "improved".
- **E5 IS PRELIMINARY — do not treat as validated.** It is a **single-chain** (the script's
  `chains=2` arg was declared and never used), **ridge=0**, **orthant-only**, **synthetic**,
  **hand-rolled** (not in-repo sampler, mass=1, no warm-up mass matrix), 2000-step run. Therefore:
  - **No R̂ and no ESS are available** (both need multiple chains), and a single trajectory
    **cannot** reveal the chain-heterogeneity/freezing that E4 showed is the dominant failure mode.
  - **The simplex at scale (which needs ridge=0.1) is entirely untested**; E5 says nothing about
    the ridge's behaviour at scale.
  - eBFMI 0.27 is *decent, not converged*.
- **TODO — revisit the at-scale (1M) setting properly** once 20×20 reproducibility is proven:
  multi-chain (R̂/ESS/per-chain eBFMI + frozen count), both regimes (simplex **with** ridge), the
  in-repo sampler with its warm-up mass matrix, and ideally the real GMVAE-derived target.

### Decision
- **The 2-week cell-level delivery can target 1000 cells/day (10⁶) — the IDEAL count — on the ridge
  sampler; Option C deferred.** (500/day = de-risked fallback.)
- Neither B nor C helps *scale* (volume term `O(m³)`); the ridge sampler is the only cell-scale tool.
- NB: this used a **hand-rolled** additive MALA; the in-repo sampler's `low_rank` path must be
  re-routed off the dead retraction branch to the generic additive path before real use.

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

---

## E11 — the gauge ridge: root cause of the convergence ceiling, confirmed at production scale
*Date: 2026-08-09 · branch: `feature/derive_volume_term`*

### Background — factorial attribution first
Every prior comparison confounded target factors. A 2×2×2 factorial (`scripts/diag_sampler_factorial.py`;
cost ∈ {real GaussianW2, E9 synthetic iid} × λ ∈ {1/0.01, 10/0.5} × Gibbs sharpness
`s = (C_max−C_min)/ε` ∈ {3.3, 33}; N=30, orthant, ridge r=2, bridge config) attributed the
real-target degradation:
- **λ-strength dominates**: warm-start `ev_max ∝ λ`; κ jumps 40–70× into the 10⁴–10⁵ range at
  λ≈10–30 — exactly the regime `λ(η)` produces for tight-marginal days. ESS drops 3–4×.
- **Cost structure is secondary** (≤2×, mostly by amplifying κ under strong λ). The E9 synthetic
  cost is concentration-flattened (contrast 0.90 vs real 3.32) — an intrinsically easy target.
- **Sharpness helps monotonically** (s=3.3→33→90 improves every pairing); the E-series' literal
  ε=0.01 sat at s≈90, its easiest corner.
- ε is not a standalone factor: it acts only through `s` (running ε=0.01 on the real cost puts
  337 nats in the exponent and overflows).

### The root cause
`κ(λ_I)` co-scaling made conditioning *worse* → the soft block is flat under **both** potential
terms → it is the **GL(r) gauge of the factor parametrisation**, unpenalised because
`ridge = 0.0` in every recorded run (E-series included; E3's caveat had flagged an unfixed gauge
for the *section* variant only). Consequences: exact flat directions, saddle warm start (46/120
non-positive Hessian directions at N=30), unbounded gauge diffusion during the unadapted warm-up
corrupting the frozen mass matrix — the universal never-strictly-converged ceiling. The ridge is
also the propriety fix: without it `S^o` is improper on the chart; with it, `−ridge·‖θ‖²` is a
proper Gaussian prior on the correction amplitude (scale to be ratified; π-bias to be quantified).

κ at the warm start (N=30, real cost, λ=10, s=33): ridge 0 → 12,324 (46 dirs ≤0);
0.1 → 105 (0 dirs ≤0); 0.5 → 8.9.

### Results — mixing follows κ
N=30 (real cost, strong-λ, orthant, bridge config):
| ridge | κ | R̂_med | R̂_max | ESS_med | ESS_min | eBFMI_min |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 12,324 | 1.62 | 2.65 | 251 | 12 | 0.164 |
| 0.1 | 105 | 1.08 | 1.40 | 459 | 24 | 0.058 |
| 0.5 | 8.9 | **1.01** | **1.10** | 810 | 143 | 0.212 |

**N=500 — the production gate** (250k-dim plans, m=2000, `max_pi_coords=2000`; ~2.3 min/run GPU):
| ridge | R̂_med | R̂_max | ESS_med | ESS_min | eBFMI_min |
|---|---:|---:|---:|---:|---:|
| 0.0 | 1.76 | 2.94 | 192 | 13 | 0.211 |
| 0.1 | 1.15 | 1.84 | 224 | 20 | 0.029 |
| 0.5 | **1.02** | **1.14** | 347 | 52 | 0.085 |

**Reproduce:** `python scripts/experiments/e11_ridge_scale.py --budget 500 --skip-hessian`
(the warm-start Hessian at m=2000 OOMs through the π-space intermediates; matrix-free Lanczos is
the follow-up for spectra at scale).

### Findings / decision
1. **The convergence ceiling was the unpenalised gauge, not MALA, not the chart family.** The
   ridge=0 control at N=500 reproduces the historical ceiling (R̂_med 1.76); ridge=0.5 reaches
   R̂_med 1.02 **on the real production-regime target** — beyond every synthetic-era result.
2. Honest residuals: eBFMI 0.085 at N=500/ridge 0.5 — energy exploration is now the weak spot
   (→ warm-up machinery: adapt-in-warm-up/freeze-for-sampling, post-transient mass window, step
   clamp; and ESJD-aware adaptation). ESS_med 347 vs 810 at N=30 — mild N-degradation, curable
   with draws. Claim precisely: *approximately converged, R̂-clean; energy mixing to harden*.
3. `ridge > 0` is **mandatory** in all future runs; scale selection + π-bias quantification is
   the open modelling item (κ(λ, ridge) is the object of the condition-number derivation).
4. The dual-potential reparametrisation is demoted from "the fix" to the scaling-optimisation
   track.

## E12 — ridge-scale ratification R2: bias grid at production scale

**Question.** How much does the gauge-breaking ridge move the published observables, and can the
gamma -> 0 limit be recovered? Grid gamma in {0.4, 0.6, 0.8, 1.2, 2.0} x {Dox D2->2.5, Serum
D12->12.5} at N=500 (bridge config x2 draws, 4 chains, thinned chart pushforward of 500
draws/chain). Observables per draw: K x K transition table (W = GMVAE posterior responsibilities
q(c|z), single-source; Schiebinger annotations cover 0% of cells before D6), R_mu/R_nu =
gKL(marginals || priors), Gibbs divergence gKL(pi || pi_I), tilt statistic T = ||theta||^2.

**Method.** Exponential-tilt identity d E_gamma[O]/d gamma = -Cov_gamma(O, T) validated against
finite differences (agreement 0.97-1.10 on all scalar segments, both phases); mean curves
near-linear in gamma, so Ehat_0[O] is taken as the quadratic-LS intercept at gamma = 0 (an
extrapolant -- the gamma = 0 chart target is improper, no chain exists there). For weakly
coupled observables (table entries) the Cov route is noise-dominated (overstates bias ~8x); the
certificate uses the grid extrapolation. Scripts: `e12_ridge_bias_grid.py` (record, incremental
per-cell checkpoint) + `e12_ridge_bias_report.py` (certificate tables + 4-panel figure).

**Certificate at gamma* = 0.40 (strengthened, threshold-free form).**
1. Location: |bias|/sd med 0.107 / max 0.231, |bias|/band max 0.058 (serum, 89/169 live entries;
   annotation-W baseline). Coverage-budget restatement: a shift of b sd erodes nominal 95%
   coverage by ~0.115 b^2 -- measured med 0.07pp / max 0.45pp against a 0.7pp budget.
2. Dual-column stability: correction is not decisive (extrapolation error ~ bias); BOTH columns
   reported and all qualitative claims invariant (Spearman 0.9998, max rank shift 1, zero rows
   change dominant destination).
3. Width NOT certified from within the family: table sd shrinks ~gamma^-0.59 with no visible
   plateau -- the small-N gold anchor (R2 step 3) is the width calibrator.
4. Structural root of the scalar blow-up (paper-grounded; HFPD-OT eq numbers): the chart
   origin theta = 0 maps to pi_I, the IDEAL design = extended Gibbs kernel (eq 5), which is
   unattainable by construction (eq 10-11). The uncentered ridge is therefore a chart-space
   surrogate for raising the ideal-attraction temperature (Remark 2): E[R_mu] rises
   ~36-40 nats per unit gamma (marginals drift from mu_0 toward pi_I's) and
   E[gKL(pi||pi^o)] rises ~35-37 (pi^o = the certainty-equivalent plan, eq 29 -- what
   `sinkhorn_init` computes; this observable was previously mislabelled g_piI). The
   contamination corrupts lambda(eta), so lambda(eta) must be solved under the ridged
   scheme; the candidate fix is a centered ridge gamma ||theta - theta_0||^2 with theta_0 =
   the exact rank-2 factorisation of pi^o ~ the S^o mode in the strong-lambda regime -- a
   quadratic penalty centred at the target mode has no first-order location effect.
   Prediction: the gKL(pi||pi^o) slope flips sign. Requires re-determination of gamma*;
   evaluated before the production campaign.

**Reproduce:** `python scripts/experiments/e12_ridge_bias_grid.py --budget 500` then
`python scripts/experiments/e12_ridge_bias_report.py`. Staging embeds on CPU (the jax pool must
not compete with torch for device memory on shared boxes).

### E12 addendum — centered ridge (2026-08-11)

Scheme change adopted after the q(c|z) certificate: the ridge is centered at the warm start,
gamma ||theta - theta_0||^2, theta_0 = the exact rank-2 factorisation of the
certainty-equivalent plan pi^o (~ the S^o mode in the strong-lambda regime; a quadratic
penalty centred at the target mode has no first-order location effect, unlike the uncentered
penalty whose lever arm |theta_bar| ~ 90 chart units pulls toward the ideal pi_I).
R1 re-determination (e11 --centered, 4 runs = both pairs x 2 chain seeds): gamma* = 0.40
re-ratified, scheme-invariant, ESS improved (466-521 vs 309-368). Centered grid
(e12 --exp cen): predictions confirmed -- g_pi0 slope flips negative, R_mu near-flat around
~3.5 nats (design reference R_mu(pi^o) ~ 0.005; the gap is the modelled coupling
uncertainty); E[T] collapses 9200 -> ~2000 (= m x posterior variance; the lever arm is
gone). Certificate at gamma*: dox PASSES cleanly (|b|/sd med 0.065 max 0.146; erosion max
0.17pp -- the diagonal-entry failure of the uncentered scheme is removed); serum med 0.070,
worst entry 0.275 with extrapolation error 0.280 (unresolved) and erosion max 0.67pp inside
the 0.7pp budget; dual-column stability holds in both phases. Mixing improves everywhere
(ESS_med 975-1843; eBFMI increases with gamma); width sensitivity halves (gamma^-0.34/-0.37
vs -0.90/-0.59). Cross-scheme gamma -> 0 intercepts agree on g_pi0 (~7.4) and correct the
uncentered long-range extrapolation of R_mu (quote ~3.4-3.6 nats, not 1.6).

**Reproduce:** `python scripts/experiments/e11_ridge_scale.py --budget 500 --ridges 0.2 0.3
0.4 0.5 --skip-hessian --centered [--chain-seed 1] [--from-day 2 --to-day 2.5]` then
`python scripts/experiments/e12_ridge_bias_grid.py --budget 500 --exp cen` and
`python scripts/experiments/e12_ridge_bias_report.py --exp cen`. Single-GPU pinning
(CUDA_VISIBLE_DEVICES) is ~20x faster than 2-GPU sharding on this host (NCCL P2P disabled:
per-step collectives serialise through the host).

### E12 addendum 2 — fixed kernel certification (2026-08-14)

Kernel frozen: centered ridge, gamma* = 0.50, windowed warm-up W = 1200 (see the warm-up A/B:
acceptance controlled at the 0.574 target with an MH-correct frozen kernel; eBFMI unmoved
across warm-up schemes => kernel-level property, E13; gamma = 0.4 passes only under the
legacy scheme's in-sampling adaptation, i.e. flattered by the correctness violation, so the
R1 rule on the fixed kernel gives 0.50). Grid re-run with 0.5 as a grid point (--exp cenw).
Certificate at gamma* = 0.50: dox passes clean (|b|/sd max 0.173, erosion max 0.27pp); serum
passes on medians (med 0.125, 0.11pp) with the worst-entry tail extrapolation-model-limited:
the mean curves carry a real, seed-reproducible non-monotone bump at small gamma (chain-seed
probe agrees to <= 0.7 MCSE on scalars), the apparent tilt-identity failure there is the Cov
noise floor at ~0.5 nats/gamma slopes, and the quad intercept's model error is comparable to
the claimed worst bias. gamma* sits near the mean curves' stationary point, so observables
are locally gamma-insensitive at the operating point. The gold anchor resolves the
consolidated docket: chart bias, width calibration, and the small-gamma intercept.

**Reproduce:** `python scripts/experiments/e12_ridge_bias_grid.py --budget 500 --exp cenw`
then `python scripts/experiments/e12_ridge_bias_report.py --exp cenw`; seed probe:
`... --ridges 0.4 0.5 0.6 --exp cenw_cs1 --chain-seed 1`.

## E13 — the gold anchor: full-rank S^o at N=8 vs the chart family

**Design.** N = 8 (plan dim 64): `full_rank` samples S^o exactly (no chart, no gauge, no
ridge -- the impropriety lived in the chart's GL(r) orbits). Both arms share the SAME MALA
schema (windowed warm-up W=1200, acceptance target, gates), differing only in the
parametrisation, so gold-vs-chart differences are attributable (no tempering; fallback
ladder is MALA-internal). Gold: 300k draws x 2 chain seeds x 4 chains. Chart arm: centered
ridge, r in {2,4,8} x gamma in {0.05, 0.1, 0.2, 0.5}. Observables as E12 (q(c|z) tables,
R_mu, R_nu, g_pi0).

**Results.** Gold impeccable: R-hat 1.00/1.00, ESS ~47k, eBFMI 0.20-0.21, seeds agree to
0.5-0.8 MCSE. Direct reads at N=8: E[R_mu] = 0.857-0.859 (both phases), E[g_pi0] = 8.63
(dox) / 14.85 (serum). Chart family: (i) R_mu bias grows more negative with gamma
(-0.43 at 0.1 to -0.60 at 0.5) and its gamma -> 0 intercept brackets (linear..quad fits on
gate-passing rows) sit at -0.36..-0.47, far from 0; (ii) E[g_pi0] shows the expected
centered-ridge bowl response (~2.4x across the ladder) riding on a gamma-INDEPENDENT ~10x
concentration gap to gold; (iii) table-entry widths are 1.1-1.65x gold at gamma=0.5,
inflating to ~2.8x as gamma -> 0, while plan-divergence spread is simultaneously ~10x
UNDER -- direction-dependent variance distortion, not a scalar width factor; (iv) the rank
effect at fixed gamma=0.2 is ~0.02 nats and not consistently monotone -- rank is not the
driver; (v) gamma = 0.05 rungs are under-mixed (R-hat_max 1.23-1.40, gauge nearly
unpinned) and are excluded from fits.

**Interpretation (hypothesis, one discriminating probe pending).** At exhaustive rank
(r = 8) with the ridge relaxed, the only remaining difference between the arms is the
reference measure: the chart targets the pullback S^o(pi(theta)) dtheta WITHOUT the
1/2 log det G volume factor; gold targets S^o(pi) dpi. The gamma-flat g_pi0 gap and the
non-closing intercepts are consistent with the missing volume term as the dominant
structural bias; the finite-gamma ladder brackets but cannot decide (the region below
gamma = 0.05 is unsampleable). DISCRIMINATOR: `low_rank_section` (Option B) carries the
exact volume term with a hard gauge and NO ridge -- no limit needed; it either closes onto
gold or refutes the attribution. Also [RESULT]: the ridge scale does not transfer across N
(gamma* = 0.5 crushes the wide N = 8 target) -- gamma is a per-N calibration.

**Figure readings.** (a) Panel-A trend (bias more negative as gamma
grows) is the centered ridge working as designed on a bowl observable: the ridge contracts
the chart posterior onto pi^o, which is gold's mode, and since R_mu is a bowl whose gold
mean is almost pure spread (E[R_mu] ~ R_mu(mode) + trace(H Sigma)/2, R_mu(pi^o) ~ 0.005 vs
gold 0.857), more contraction pushes E_chart further below gold. The figure omits
gate-failing rungs from panel A (their only content is panel D's story) and draws the
gamma -> 0 intercepts as linear..quad brackets -- a plausible range, not a point estimate.
(b) Panel-C anisotropy has one mechanism with both signs: g_pi0 = KL(pi || pi^o)
accumulates variance ADDITIVELY (a bowl in every coordinate, no cancellation), so gold's 64
near-independent fluctuations sum to a large KL radius the rank-r chart structurally caps
(~10x under). A table entry is a SIGNED weighted aggregate: independent fluctuations
partially cancel inside it, but the chart's moves perturb whole rows/columns of log pi
coherently (one factor coordinate at a time) and coherent moves add in phase -- so the
chart's smaller total variance is concentrated on exactly the directions table aggregates
amplify (1.1-2.8x over). The chart redistributes covariance rather than shrinking it
uniformly; no scalar width factor can correct both projections, and D2 held-out coverage
remains the end-to-end audit for this anisotropy at production scale.

**TODO (SIAM track, deliberately deferred):** run the `low_rank_section` discriminator
(implemented in `supports.py`: pivoted hard gauge, exact 1/2 log det G_S by autodiff,
ridge = 0) at N = 8 against the gold record -- the constructive test of the volume-term
attribution. Off the production critical path: the Nature pipeline proceeds on the
certified fixed kernel.

**Reproduce:** `python scripts/experiments/e13_gold_anchor.py --budget 8 --exp gold` then
`python scripts/experiments/e13_gold_report.py --exp gold`.
