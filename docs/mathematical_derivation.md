# Low-rank HFPD-OT sampling — mathematical foundations

This note describes the math behind the **log-low-rank** reparametrization for
sampling high-dimensional HFPD-OT transport plans. It is the reference for the
implementation in `src/`.

Status: living document. Point 1 (the volume factor) is settled below; point 2 (gauge
fixing) and point 3 (matrix-free determinant) are WIP.

---

## 0. Notation and setup

- Plan dimensions `II × JJ`; ambient dimension `n = II·JJ` (target ~10^6).
- Gibbs kernel `K = exp(-C/ε)`, `C` the cost, `ε` the entropic regularization.
- **Parameters** `θ = (U, V)`, `U ∈ ℝ^{II×r}`, `V ∈ ℝ^{JJ×r}`, rank `r ≪ min(II,JJ)`.
- **Reconstruction** (the two-layer map):

      θ  --(bilinear, low-rank)-->  ℓ = U Vᵀ − C/ε  =  log π   --(exp, = PositiveOrthant, softmax, = Probability Simplex)-->  π
  i.e. `π_ij = exp( (UVᵀ)_ij − C_ij/ε )` or `π_ij = softmax( (UVᵀ)_ij − C_ij/ε )`.

We write: `Φ := (softmax/exp) ∘ (θ ↦ UVᵀ − C/ε)` the composite map `θ → π`.

- The inner layer `ℓ ↦ π = exp(ℓ)` is exactly our tested `PositiveOrthant` support
  (`to_constrained = exp`, `log_det = Σℓ`, `latent_score = π⊙s + 1`) -- `s` being our target hyperprior score function.
- The outer layer `θ ↦ ℓ` is new, and is **linear in ℓ** (bilinear in θ).
- **The dual potentials case is the r = 2 special case**: `U = [f/ε, 𝟙]`, `V = [𝟙, g/ε]` gives
  `ℓ_ij = (f_i + g_j − C_ij)/ε`, the entropic-OT dual potentials (Sinkhorn). Use this both
  as a warm start and as a correctness oracle.
- **Raw parameter count** `dim θ = r(II + JJ)` — what MALA actually samples. The `GL(r)`
  gauge (§2) has dimension `r²`, so the **intrinsic dimension** of the image manifold `M_Φ`
  — the Hausdorff dimension used by `ℋ^m` and `G` throughout — is
  `m := dim M_Φ = r(II + JJ) − r²`, reached by the **canonical balanced gauge** fix. For
  `II=JJ=10³, r=5`: `dim θ ≈ 10⁴`, `m = dim θ − 25`, vs `n = 10⁶`.

---

## 1. The target density and the volume factor `½ log det G`

### 1.1 Why this is NOT ordinary change-of-variables

Ordinary change-of-variables (the `|det J|` formula) requires a **diffeomorphism between
spaces of equal dimension**: `J` square, `q(θ) = p(φ(θ))·|det J|`. That is the classic regime encoded in
`supports.py` (e.g. orthant `y∈ℝⁿ → x∈ℝⁿ`, `log|det J| = Σy`).

The low-rank transformation is different, and the reason is twofold:

1. **Rectangular Jacobian.** The Jacobian of the map `Φ`: `J = DΦ` is
   `n×m` (tall). A non-square matrix has **no determinant** — `|det J|` is undefined.
2. **Measure-zero image.** `M_Φ = Φ(ℝ^m) ⊂ ℝⁿ` is a thin `m`-dimensional submanifold of the *ambient* `ℝⁿ` and, since `m < n`, is **Lebesgue-null in `ℝⁿ`**. Even though each `π` is full matrix rank (the Hadamard exponential destroys the rank‑`r` structure of a single plan), the *family* `M_Φ` has only as many degrees of freedom as `θ`, namely `m`.

We are not transporting a density across a bijection; we are **restricting** a density to `M_Φ`. Restriction to a null set is not change-of-variables — it requires *choosing a reference measure on `M_Φ`*. The canonical choice is the `m`-dim **Hausdorff measure** `ℋ^m`.

### 1.2 The area formula

Write the **Jacobian factor** `J_Φ(θ) := √det( DΦ(θ)ᵀ DΦ(θ) )`. For **any** Lipschitz
`Φ: ℝ^m → ℝ^n` with `m ≤ n`, the area formula of geometric measure theory (Federer 1969,
Thm 3.2.3; Evans–Gariepy 2015, §3.3) states, for every integrable `g` on `ℝ^m`,

    ∫_{ℝ^m} g(θ) J_Φ(θ) dθ  =  ∫_{ℝ^n} ( Σ_{θ ∈ Φ⁻¹(y)} g(θ) ) dℋ^m(y).      (general)

The inner sum runs over the **fibre** `Φ⁻¹(y)`; its size is the counting measure
`N(y) := ℋ⁰(Φ⁻¹(y))` — the multiplicity with which `Φ` covers `y`. This multiplicity is the
term that must be tracked; it is *not* `ℋ^m` of the fibre.

We only ever integrate a function *of the plan*, so specialize `g = f∘Φ` for a test function
`f` on the image. Then `g(θ) = f(y)` for every `θ` in the fibre of `y`, the inner sum becomes
`f(y)·N(y)`, and

    ∫_{ℝ^m} f(Φ(θ)) J_Φ(θ) dθ  =  ∫_{ℝ^n} f(y) · N(y) dℋ^m(y).                (pulled back)

Two assumptions of our problem now collapse `N`:

- **Gauge-fixing ⇒ injectivity (§2).** The raw map is *not* injective — the `GL(r)` gauge
  `(U,V)→(UR,VR⁻ᵀ)` makes each fibre an `r²`-dimensional orbit, so `N ≡ ∞`, `DΦ` is
  rank-deficient, `J_Φ = 0` a.e., and *(pulled back)* degenerates to `0 = 0`. On the
  gauge-fixed section `Φ` is injective, so `N(y) = 1` on `M_Φ` and `0` off it, i.e.
  `N ≡ 1_{M_Φ}`.

Substituting `N ≡ 1_{M_Φ}` gives the identity we actually use:

    ∫_{M_Φ} f dℋ^m  =  ∫_{ℝ^m} f(Φ(θ)) J_Φ(θ) dθ.                             (ours)

The columns of `J = DΦ` are the `m` tangent vectors `∂Φ/∂θ_k`. `G := JᵀJ` is their **Gram
matrix**, and `√det G` is the **`m`-volume of the parallelepiped** they span in the ambient
space (= product of the `m` singular values of `J`). This is the right "stretch factor" for
a map that cannot fill the ambient space.

### 1.3 It generalizes — not replaces — `|det J|`

When `m = n` (square `J`), `det(JᵀJ) = (det J)²`, so `√det(JᵀJ) = |det J|`. The orthant
support is the `m=n` corner of the *same* formula. Hence

    ½ log det G   =   log √det(JᵀJ)   =   the honest extension of  log|det J|
                                          to a dimension-reducing map.

### 1.4 The target density on θ

`M_S`, the support of the HFPD-OT hyper-prior, is full-dimensional. On the other hand, and as per §1.1, `M_Φ` is Lebesgue-null. This has important implications on the uncertainty representation power of `π` (subtle, will be fully discussed separately).

Restricting the HFPD-OT posterior `p(π)` to `M_Φ` w.r.t. `ℋ^m`, pulled back to θ:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  log q(θ) = log p( π(θ) )  +  ½ log det G(θ)  +  const               │
    │            G = JᵀJ ,   J = ∂(π(θ))/∂θ_free                       │
    └─────────────────────────────────────────────────────────────────────┘

- `θ_free` are the **canonical balanced gauge** coordinates (§2): the `m = r(II+JJ) − r²`
  directions transverse to the `GL(r)` orbit. This is load-bearing — on the raw sampled
  `θ ∈ ℝ^{r(II+JJ)}` the orbit lies in `ker J`, so `G = JᵀJ` is rank-deficient by exactly
  `r²`, `det G = 0`, and `½ log det G = −∞`. The volume term is well-defined **only** on the
  section; §2 (Faddeev–Popov) is how sampling on the raw `θ` recovers it — and why v1's ridge
  is a stopgap for the missing term.
- `log p(π(θ))` is the existing HFPD-OT hyperprior (balanced shifted-KL on the proba. simplex,
  or unbalanced generalized-KL on the pos. orthant), evaluated at the reconstructed plan.
- `½ log det G` is the **induced-volume (Riemannian) Jacobian** from §1.2.

### 1.5 Is `½ log det G` really a "Jeffreys" prior?

An important subtlety to clear here: `½ log det G` is **not** literally `½ log det(Fisher)` in general — calling it "Jeffreys" would not be correct. More precisely:

- For *any* Riemannian metric `g`, `√det g · dθ` is the **invariant Riemannian volume
  measure** (reparametrization-invariant: under `θ→φ`, `g→JᵀgJ`, so `√det g` picks up
  `|det J|`).
- The **Jeffreys prior** is the special case `g = 𝓘(θ)` (Fisher information).
- Our `G = JᵀJ` is the metric the **ambient Euclidean geometry** induces by pullback — a
  priori not a Fisher information.

The bridge that earns the name: posit an **isotropic-Gaussian embedding model**,
`data ∼ 𝒩(Φ(θ), σ²I)` in the ambient space. Its Fisher information is
`𝓘(θ) = (1/σ²) DΦᵀDΦ = G/σ²`, so `√det G ∝ √det 𝓘`. Thus `½ log det G` **is** the Jeffreys
prior of the isotropic-Gaussian embedding model.

**Terminology going forward:** call it the *induced-volume (Riemannian) Jacobian*, noting it
coincides with Jeffreys for the isotropic-Gaussian embedding metric.

### 1.6 Modeling caveat — restriction vs marginalization, and the off-manifold dispersion

Because `M` -- the support of the HFPDOT hyperprior -- is measure-zero, "restrict to `M` w.r.t. `ℋ^m`" is a **modeling choice**, not a
theorem. It is the canonical invariant choice, but it is **not** the same distribution as
"marginalize the full ambient posterior onto `M`-coordinates."

- **Pure restriction.** Off-manifold fluctuations are set to zero. Captures only the
  dispersion *along* `M` (i.e. uncertainty in the marginals + whatever rank-`r` off-manifold
  directions we explicitly sample). This is exact in the `ε→0` / sharp-`K` limit, where the
  posterior really does collapse onto `M`.

- **Normal-bundle Laplace marginalization (future).** At each `π(θ)∈M`, split the ambient tangent space into the `m` directions *tangent to `M`*
and the `n−m` directions *normal to `M`*. Integrate the full posterior over the normal
directions under a **local Gaussian (Laplace) approximation** — a collar of finite width
hugging `M`:

      ∫ p(π(θ) + N z) dz  ≈  p(π(θ)) · (2π)^{(n−m)/2} / √det H_⊥(θ),
      H_⊥(θ) = Nᵀ [ −∇²log p(π(θ)) ] N     (off-manifold Hessian; N = orthonormal normal basis)

  giving an effective on-manifold target

      log q_tube(θ) = log p(π(θ)) + ½ log det G(θ) − ½ log det H_⊥(θ) + const.

  The collar **width is not a free parameter** — it is the posterior's own normal-direction
  curvature, i.e. the genuine off-manifold variance to second order. So it *recovers the
  off-manifold uncertainty that pure restriction discards, without raising `r`*. Cost: another
  (larger, `(n−m)`-dim) log-det `det H_⊥`, which is why it is a v2 refinement, not v1.

  **Collar vs. raising `r` are complementary knobs.** Raising `r` *moves* directions from the
  normal (integrated) bundle into the tangent (explicitly sampled) bundle; the collar
  *Gaussian-approximates* whatever stays normal. v1 = pure restriction at small `r`; later we
  can either raise `r` or add the collar (or both) to restore off-manifold dispersion.

---

## 2. Gauge fixing (the `GL(r)` non-identifiability)

### 2.1 The symmetry

For any invertible `R ∈ GL(r)`,

    U' = U R,   V' = V R⁻ᵀ   ⟹   U'V'ᵀ = U R R⁻¹ Vᵀ = U Vᵀ,

so `ℓ`, `π`, and the likelihood are **exactly constant** on the orbit
`{(UR, V R⁻ᵀ) : R ∈ GL(r)}`. Each orbit is an **`r²`-dimensional** flat ridge through `θ`.

### 2.2 Why this is not optional — it breaks §1 immediately

The orbit tangent directions lie in **`ker J_ℓ`** (moving along an orbit does not change `ℓ`).
Hence `G = J_ℓᵀ J_ℓ` is **rank-deficient by exactly `r²`**:

    rank G = r(II+JJ) − r²,    det G = 0,    ½ log det G = −∞.

The §1 volume term is *undefined* until the gauge is handled — so gauge fixing should be addressed before
the volume term derivation.

### 2.3 Why the Riemanian volume factor cannot fix it alone

`GL(r)` is **non-compact**, so every orbit has **infinite volume** (`R` can stretch/shrink
without bound). A volume/Jeffreys term measures the **quotient** `M = Θ/GL(r)` — the geometry
of the *base*. It says nothing about the **fibre** (the orbit). No quotient volume form can
integrate an infinite fibre. Sampling `p(π(θ))` over the full `θ`-space leaves the target flat
along each orbit → **improper** posterior; the chain random-walks along the `r²`-dim ridge and
`U,V` summaries are meaningless.

These are **orthogonal jobs**, and we need both:
- **gauge fixing** → properness (handles the non-compact orbits);
- **volume term** → coordinate-invariant measure (handles the quotient geometry).

### 2.4 The fix: restrict to a section

Pick `S ⊂ Θ` transversal to the orbits, in bijection with `M`. Two choices:

**(i) Hard fixed gauge — fix the top `r×r` block of `U` to `I_r` [not recommended].**
Given `(U,V)` with `U_top` invertible, set `R = U_top⁻¹`; then `U' = UR` has `U'_top = I_r`.
Free coordinates: `U_bot ∈ ℝ^{(II−r)×r}`, `V ∈ ℝ^{JJ×r}`, dimension `r(II+JJ) − r²` exactly.
A **flat Euclidean chart** (delete `r²` coordinates) — trivial for MALA/HMC, no manifold
machinery.
- Fixes the gauge **completely, continuous *and* discrete**: from the target `L = UVᵀ`,
  `U_top = I_r` forces `Vᵀ = L_top`, then `U = L V (VᵀV)⁻¹` — a unique representative, no
  residual sign/permutation freedom.
- Limitation: valid only where `U_top` is invertible (a *chart*, not global). A wandering
  chain can hit `U_top → singular`; mitigate by pivoting (pin `r` independent rows) or by (ii).
- Enforces an assymetric -- unnatural -- structure in the parameter space, where V may grow substantially to balance out the fixed `rxr` block in `U_top`.

**(ii) Balanced canonical gauge — `UᵀU = VᵀV` diagonal [robust, implemented].** The SVD-flavoured
symmetric form; globally better-conditioned but needs Stiefel-style constrained handling
(toward Option C).

### 2.5 No separate Faddeev–Popov determinant in the direct parametrization

Two routes reach the **same** measure on `M`. The **factor-out** route (Faddeev & Popov,
*Phys. Lett. B* **25** (1967) 29–30) starts from the full-`θ` integral,
`∫_Θ = (orbit volume) × ∫_S (FP det)(·)`, and carries a Faddeev–Popov determinant for the
gauge condition. The **direct** route (what we implement) never forms that integral: it
parametrizes `M` straight from the section `S` and reads the target off the area formula,
`p(π) · √det G_S`, with `G_S` full-rank (the `r²` kernel directions are gone). The FP
determinant is not a *separate* object — it is constant for the hard gauge and already inside
`√det G_S` for the balanced gauge. We are free to chart the base directly (unlike a field
theory, whose field lives in the redundant space), so we take the direct route.

So in code: **gauge fix = sample on the section + evaluate `½ log det G_S`. One determinant.**

### 2.6 A free correctness test

The target *measure on `M`* (hence the `π`-marginals) is **section-independent** — it is the
intrinsic Hausdorff measure `ℋ^m`. Different valid sections give different coordinates and
different MCMC efficiency but the **same sampled distribution on `π`**. ⟹ **gauge-invariance
test**: sample under chart (i) and under a perturbed/alternative section; the `π`-marginals
must match. (Poworoznek, Ferrari & Dunson 2025's Varimax+matching post-processing is the
factor-analysis community's *alternative* to hard fixing — a fallback if chart (i)'s
singularities bite.)

## 3. Which uncertainty the sampler explores — core vs subspace

Fixing the rank `r` (the low-rank restriction, §1) and the gauge (§2) is not enough: the
tangent space of `M_Φ` splits into two geometrically distinct blocks, and a proposal that
touches only one of them samples only half the posterior. This section is what makes the
*proposal design* (§3.4, Option B/C below) a correctness issue, not just an efficiency one.

### 3.1 The tangent decomposition

Parametrize a plan by `(U,V)`, `U ∈ ℝ^{II×r}`, `V ∈ ℝ^{JJ×r}`. Any infinitesimal move splits
uniquely into a part inside the current column space and a part orthogonal to it:

    δU  =   U a          +      U_perp c
            (in col U)          (⊥ col U)
    δV  =   V b          +      V_perp d

- **core** `(a, b) ∈ ℝ^{r×r} × ℝ^{r×r}` — recombine / rescale the existing `r` columns among
  themselves; `col(U)`, `col(V)` are **unchanged**. Dimension `2r²`.
- **subspace** `(c, d)` — inject directions *orthogonal* to the current columns; this **tilts**
  `col(U)`, `col(V)`. Dimension `r(II−r) + r(JJ−r) = r(II+JJ) − 2r²`.

The `GL(r)` gauge (§2.1) lives **entirely inside the core** — `(Uξ, −Vξᵀ)`, `ξ ∈ gl(r)`,
dimension `r²`. Quotient it out and the physical manifold `M_Φ` (dim `r(II+JJ) − r²`, §0) is

    dim M_Φ  =    r²              +    [ r(II+JJ) − 2r² ]
                  └ physical core      └ subspace (two Grassmannians Gr(r,II) × Gr(r,JJ))

For `II=JJ=20, r=2`:  `4 + 72 = 76`.

### 3.2 What each block *is* — the SVD reading

Write `U Vᵀ = Σ_k σ_k û_k v̂_kᵀ`. Then:

- **subspace** = the singular *vectors* `û_k, v̂_k` — the **shapes** of the `r` coupling modes
  (which sources couple to which targets);
- **core** = the singular *values* `σ_k` plus the `r×r` alignment — the **strengths** of those
  modes.

Freezing the subspace = fixing the singular vectors from the warm start and sampling only the
singular values.

### 3.3 The uncertainty consequence

Posterior uncertainty in a plan is therefore two distinct things:

- **strength UQ (core):** given the modes, how much mass flows through each;
- **structure UQ (subspace):** what the modes *are* — the coupling pattern itself.

These are not equally important for the application. In single-cell lineage inference the
**subspace is the science** (which progenitor populations map to which descendants); the core
is only the flow magnitude along an *assumed* structure. A proposal that freezes the subspace
reports **zero structural uncertainty** and falsely tight credible bands — it is confidently
wrong about precisely the quantity the method exists to estimate.

This is a **second, nested restriction**, on top of the low-rank one:

    full plans (ℝⁿ)  ⊃  M_Φ (rank-r log)  ⊃  core-slice (frozen subspace)
                        └ low-rank restriction (§1)     └ this section

The first is a deliberate modeling choice with a knob (`r`); the second must **never** be
incurred *accidentally* by the choice of proposal.

### 3.4 Design consequence — admissibility of a proposal

A right-multiplicative retraction `U ↦ U · exp(Ω_U)` with `Ω_U ∈ ℝ^{r×r}` is **pure core**
(`c = d ≡ 0`): right-multiplication cannot change a column space, so `col(U_k) = col(U_0)` for
the *entire* chain. It explores `r²` of the `r(II+JJ) − r²` physical dimensions (4 of 76 above)
and freezes the subspace — hence samples strength UQ only.

**Admissibility rule: a proposal must move the subspace block `(c, d)`.** Two that do:

- **Option B (flat chart).** Sample the hard-gauge section (`U_top = I_r`, free `U_bot, V`)
  with *additive* MALA plus the volume term `½ log det G_S`. Additive steps on `U_bot` tilt
  `col(U)`, so the subspace moves; the state is flat coordinates, so there is no manifold to
  float off and no retraction is needed. Simpler; the natural home for the §1 volume term.
- **Option C (Riemannian).** Keep a *constrained* (balanced-canonical / Stiefel) state and use
  a **subspace-moving** Stiefel retraction (QR / Cayley) — *not* the core-only `exp`. Better
  conditioned, more machinery (a Stiefel-tangent Jacobian on top of the volume term).

The core-only exponential retraction is **inadmissible** for full-manifold UQ. We ship **B**
first (does it move eBFMI at all?), and investigate **C** as the rigorous next step.

## 4. Matrix-free determinant (the scaling crux) — TODO

Stub. To cover: exact `log det G` is `O(min(II,JJ)³ r³)/step` → defeats scaling; never form
`G`, only `G·v` at `O(II·JJ·r)`; `log det G` via stochastic Lanczos quadrature; `∇ log det G
= tr(G⁻¹∂G)` via Hutchinson + CG; validate against exact `slogdet` at small scale.

---

## References

- Jeffreys (1946), *An invariant form for the prior probability in estimation problems*,
  Proc. Roy. Soc. A 186. — `√det(Fisher)` and its invariance.
- Diaconis, Holmes & Shahshahani (2013), *Sampling From A Manifold*, arXiv:1206.6913. — the
  area formula and `√det(JᵀJ)` densities on embedded submanifolds.
- Byrne & Girolami (2013), *Geodesic Monte Carlo on Embedded Manifolds*, Scand. J. Stat.
  40(4), arXiv:1301.6064. — MCMC on embedded manifolds incl. Stiefel (Option C).
- Girolami & Calderhead (2011), *Riemann manifold Langevin and HMC methods*, JRSS-B 73(2). —
  sampling with a metric; `log det(metric)` in the dynamics.
- Edelman, Arias & Smith (1998), *The geometry of algorithms with orthogonality constraints*,
  SIMAX 20(2):303–353. — Stiefel/Grassmann tangent geometry and retractions (§3 core-vs-
  subspace; Option C).
- Faddeev & Popov (1967), *Feynman diagrams for the Yang–Mills field*, Phys. Lett. B 25. —
  gauge-fixing + orbit-volume determinant (for §2).
- Poworoznek, Ferrari & Dunson (2025), *Efficiently Resolving Rotational Ambiguity in
  Bayesian Matrix Sampling with Matching*, Bayesian Analysis, arXiv:2107.13783. — the
  statistical analog of the gauge (for §2).
- Ubaru, Chen & Saad (2017), *Fast Estimation of tr(f(A)) via Stochastic Lanczos
  Quadrature*, SIMAX 38(4):1075–1099. — matrix-free `log det` / `tr(G⁻¹·)` (for §3).
- Hutchinson (1990), *A stochastic estimator of the trace of the influence matrix*,
  Comm. Stat. Simul. 19(2). — the probe-vector trace estimator (for §3).
- Cuturi (2013), *Sinkhorn Distances*, NeurIPS; Peyré & Cuturi (2019), *Computational
  Optimal Transport*, FnT ML. — entropic-OT dual potentials (the r=2 case A).
- Boufelja Y., Quinn & Shorten (2025), *Randomized transport plans via HFPD*,
  Information Sciences. — the HFPD-OT model; Algorithm 1 (QN on potentials) = A.
