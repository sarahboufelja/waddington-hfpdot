# Option C — balanced-canonical Stiefel-manifold sampler (implementation proposal)

Status: **locked proposal** / design doc. Companion to
[`mathematical_derivation.md`](mathematical_derivation.md) (theory) and
[`experiments.md`](experiments.md) (E1–E3, Option B).

## 0. Why C, and what it is / isn't for

Option B (hard-gauge flat chart + volume term) works but is **asymmetric and stiff**: pinning
`U_top=I_r` forces scale into `V`, and neither diagonal nor mode-Hessian preconditioning can
handle the position-dependent curvature. Option C replaces the flat chart with the **balanced
canonical gauge** — the *symmetric*, well-conditioned, fully-gauge-fixed parametrization — at
the cost of a **curved** (Stiefel) state that needs a retraction.

### 0.1 Empirical motivation (E1–E3)
- **E3 (full 20×20):** Option B is a *mixing regression* vs the shipped ridge — orthant eBFMI
  **0.067 vs 0.31**, ESS ~10 vs 899; and it *degrades with dimension* (10×10 eBFMI 0.21 → 20×20
  0.067). The volume term is the correct measure, but the hard-gauge geometry is unsamplable-well.
- So C is motivated by data, not just theory: we need the **symmetric** geometry to make the
  measure-correct target actually mix.

### 0.2 Scope discipline (the axis separation)
- C targets **mixing quality / conditioning**, not scale.
- **Neither B nor C scales to 10⁶ without the §4 matrix-free log-det.** The exact-autodiff
  volume term is `O(n·m + m³)`/step — fine at `n=400`, impossible at `n=10⁶`. C is the
  well-conditioned geometry that a matrix-free volume term then makes scalable.

## 1. Parametrization — the SVD / balanced canonical form

Represent the log-plan `L = UVᵀ` (`π = to_constrained(vec(L) − C/ε)`) by its thin SVD:

    L = P diag(σ) Qᵀ,     P ∈ St(r, II),  Q ∈ St(r, JJ),  σ ∈ ℝ^r_(>0)

where `St(r,n) = { X ∈ ℝ^{n×r} : XᵀX = I_r }` (the Stiefel manifold). This *is* the balanced
gauge: `U = P diag(√σ)`, `V = Q diag(√σ)`, so `UᵀU = VᵀV = diag(σ)` — symmetric between the two
factors, no `inv(U_top)` blow-up.

**Dimension count (exactly `dim M_Φ`):**

    dim St(r,II) + dim St(r,JJ) + dim(σ)
      = [II·r − r(r+1)/2] + [JJ·r − r(r+1)/2] + r
      = r(II+JJ) − r²                                  ✓ = dim M_Φ

The `GL(r)` gauge is fully fixed; the only residue is the discrete sign/permutation of singular
triples (measure-zero; handled by ordering σ descending).

### 1.1 The simplex carries a SECOND gauge the Stiefel form does not fix

The `GL(r)` gauge above is a redundancy of the **factorization** layer `(U,V)→ℓ`. The
**simplex** carries an *independent* redundancy of the **softmax** layer `ℓ→π`: the
softmax-shift `softmax(ℓ + c·𝟙) = softmax(ℓ)`. For `r≥2` the low-rank family can represent the
rank-1 constant `𝟙𝟙ᵀ`, so this shift is a (near-)flat direction of the parameter→π map (`G=JᵀJ`
goes near-singular along it) — and it is **orthogonal to, and untouched by, both the hard-gauge
section (Option B) and the Stiefel parametrization**. This is exactly why v1 needed a ridge for
the simplex but not the orthant. **Required for the simplex regime:** C must add a *separate*
shift fix — a proper prior on `Σℓ` (mirroring the full-rank `Simplex` support's radial
coordinate) or an explicit shift coordinate — otherwise the simplex manifold is incompletely
gauge-fixed regardless of the factorization geometry. The **orthant** (exp map) has **no** such
gauge (`exp(ℓ+c)` rescales mass, which the unbalanced hyperprior penalizes), so it needs only
the `GL(r)` fix and is the clean regime for benchmarking.

## 2. Geometry primitives (per Stiefel factor `X ∈ St(r,n)`)

- **Tangent space:** `T_X = { ξ : Xᵀξ + ξᵀX = 0 }` (`Xᵀξ` skew). Dimension `nr − r(r+1)/2`.
- **Projection** of ambient `Z ∈ ℝ^{n×r}` onto `T_X` (Euclidean metric):

      Proj_X(Z) = Z − X · sym(Xᵀ Z),     sym(A) = (A + Aᵀ)/2

- **Retraction** (tangent → manifold), QR-based (cheapest, `O(n r²)`):

      Retr_X(ξ) = qf(X + ξ)     [ qf = Q-factor of QR, positive diagonal ]

  Cayley or geodesic (matrix-exp) retractions are drop-in alternatives if QR is unstable.
- **Riemannian gradient:** `rgrad f = Proj_X(∇_X f)`.

**Crucially, unlike the broken core-only `U·exp(Ω)` retraction, a Stiefel tangent step moves the
column space** (`ξ` has a component in `X_⊥`), so it explores the subspace — the property §3 of
the derivation showed is mandatory for structural UQ.

## 3. The proposal and the MH correction (the hard part)

Target on the parameter manifold `N = St(r,II) × St(r,JJ) × ℝ^r`:

    log q(P,Q,σ) = log p(π(P,Q,σ)) + ½ log det G_C(P,Q,σ)

(`G_C` = Gram of `∂π/∂(tangent of N)` — see §4). The volume term is **still required**: the
Hausdorff measure `ℋ^m` on `M_Φ` and the Riemannian volume of `N` differ by exactly this
Jacobian, and consistency with B means C targets the same `ℋ^m`.

Two routes for the kernel:

**(C-i) Retraction MALA** — propose `ξ = (τ/2)·rgrad + √τ·Proj_X(Z)`, `Z` Gaussian; retract each
factor; Euclidean step on `σ`. **MH bookkeeping is the cost:** the retraction is not symmetric,
so the reverse density needs (a) the reverse tangent — invert the retraction — and (b) the
retraction's log-Jacobian. This is the structure the *deleted* balanced-gauge code was reaching
for, but now with the **correct, subspace-moving** retraction. Risk: inverse-retraction +
Jacobian is fiddly and error-prone (exactly where the old code had bugs).

**(C-ii) Geodesic Monte Carlo / HMC** (Byrne & Girolami 2013, already cited) — evolve along
Stiefel **geodesics** with a volume-preserving, reversible integrator. Reversibility is *by
construction*, so **no explicit retraction-Jacobian** in the accept step. More upfront work
(geodesic flow + constrained leapfrog), but far more robust and the natural paper-grade choice.
**Recommended.**

## 4. The volume term on the Stiefel tangent

`G_C = J_Tᵀ J_T`, where `J_T = ∂π/∂θ` restricted to an orthonormal basis of `T_N` (Stiefel
tangents ⊕ ℝ^r). Pick a basis `{e_k}` of `T_N` (dim `m`), form `J_T = [ ∂π·e_k ]` (`n×m`), then
`½ log det(J_Tᵀ J_T)`.
- **Small scale:** exact autodiff (jvp along the basis) + `slogdet`, as in B.
- **Scale (§4 of the derivation):** never form `G_C`; stochastic Lanczos quadrature for
  `log det`, Hutchinson + CG for `∇ log det` — the *shared* scaling blocker with B.

## 5. Phased implementation plan

- **C0 — geometry primitives.** `stiefel_proj`, `stiefel_retract` (QR), `stiefel_tangent_basis`,
  Riemannian grad. Tests: `XᵀX=I` preserved under retraction; projection idempotent + lands in
  tangent; SVD ↔ `(P,σ,Q)` round-trip.
- **C1 — a `StiefelLowRank` support** exposing `(P,Q,σ) ↔ π`, warm-started from the Sinkhorn SVD.
  Reuse B's isolated-verification harness (subspace-motion, round-trip).
- **C2 — the kernel.** Start with **C-ii geodesic HMC** (reversible, no Jacobian); fall back to
  C-i retraction-MALA only if HMC integration is too costly. Validate: reversibility numerical
  check; `r=2` Sinkhorn oracle; healthy acceptance; **subspace moves**.
- **C3 — volume term** `½ log det G_C` (autodiff first, matrix-free per §4 later).
- **C4 — head-to-head** vs the ridge + B on the 20×20 bridge (eBFMI, R̂, ESS), then the
  **high-dim stress test** (`II=JJ~10³`): run/compile/memory, per-step cost breakdown, mixing
  survival.

## 6. Open questions to resolve before coding
1. HMC (C-ii) vs retraction-MALA (C-i) — commit to HMC unless integration cost forbids it?
2. Keep the exact `ℋ^m` volume term, or (modeling choice, §1.6) accept the Stiefel Riemannian
   volume as the target measure and **drop `G_C`**? The latter removes the hardest piece but
   changes the sampled measure — decide, and document as a variant.
3. Metric on `σ` (log-σ vs σ) — log-σ keeps positivity and is scale-natural.
