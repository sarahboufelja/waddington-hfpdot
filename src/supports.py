r"""Constrained-support reparametrizations for Langevin sampling.

The Langevin sampler explores an *unconstrained* latent state :math:`y \in \mathbb{R}^d`.
The target distribution, however, may live on a constrained set: the probability
simplex (for balanced HFPD-OT plans) or the positive orthant (for unbalanced
plans). Each :class:`Support` owns a single, self-consistent change of variables:

* the forward map ``to_constrained`` :math:`y \mapsto x`,
* its inverse ``to_unconstrained`` :math:`x \mapsto y`,
* the log-Jacobian-determinant ``log_det_jacobian`` of that map, and
* the lifted target ``latent_log_prob`` / ``latent_score`` on :math:`y`.

The target log-density and score are supplied in the **constrained** coordinates
(natural for the modeller); the support is solely responsible for the chain rule
and the Jacobian term. By construction ``latent_score`` is the analytic gradient
of ``latent_log_prob`` -- the test-suite asserts this against ``jax.grad``.

Notation. For a callable ``f`` we write ``f`` for the target log-density on the
constrained space and ``s = grad f`` for its score (gradient w.r.t. the
constrained coordinate).
"""

from abc import ABC
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp
import jax.scipy.stats as jstats

LogProbFn = Callable[[Array], Array]
ScoreFn = Callable[[Array], Array]

@dataclass
class NoiseMatrices:
    A_noise_U: Array
    A_noise_V: Array
    S_noise: Array

@dataclass
class BalancedGaugeDrift:
    shared_symmetric_part: Array
    assymetric_u: Array
    assymetric_v: Array
    
@dataclass
class StateSummary:
    """
    Summary of a proposed state in the balanced gauge constraint manifold.
    Returns the proposed state, the noise matrices that gave rise to the proposal, and the velocities used in the retraction step.
    """
    state: Array
    balanced_gauge_drift: Optional[BalancedGaugeDrift] = None
    matching_noise: Optional[NoiseMatrices] = None
    velocities: Optional[Array] = None

class Support(ABC):
    """Abstract base class for a constrained-support reparametrization.

    Subclasses operate on the trailing axis of ``y`` (one state per leading
    index). ``latent_log_prob`` returns a scalar (per state) and
    ``latent_score`` returns an array shaped like ``y``.
    """

    name: str = "support"

    def to_constrained(self, y: Array) -> Array:
        """Map an unconstrained latent state ``y`` to the constrained space."""
        raise NotImplementedError

    def to_unconstrained(self, x: Array) -> Array:
        """Map a constrained state ``x`` back to the latent space."""
        raise NotImplementedError

    def log_det_jacobian(self, y: Array) -> Array:
        r"""Return :math:`\log |\det \partial x / \partial y|` at ``y``."""
        raise NotImplementedError

    def latent_log_prob(self, target_log_prob: LogProbFn, y: Array) -> Array:
        """Lift a constrained-space log-density to the latent space."""
        raise NotImplementedError

    def latent_score(self, target_log_prob: LogProbFn, target_score: ScoreFn, y: Array) -> Array:
        """Lift a constrained-space score to the latent space (gradient of
        :meth:`latent_log_prob`).

        ``target_log_prob`` is accepted for signature symmetry; the analytic
        lift only needs the constrained-space ``target_score``.
        """
        raise NotImplementedError


class Unconstrained(Support):
    r"""Trivial support: the target already lives on :math:`\mathbb{R}^d`."""

    name = "unconstrained"

    def to_constrained(self, y: Array) -> Array:
        return y

    def to_unconstrained(self, x: Array) -> Array:
        return x

    def log_det_jacobian(self, y: Array) -> Array:
        return jnp.zeros(y.shape[:-1])

    def latent_log_prob(self, target_log_prob: LogProbFn, y: Array) -> Array:
        return target_log_prob(y)

    def latent_score(self, target_score: ScoreFn, y: Array) -> Array:
        return target_score(y)


class PositiveOrthant(Support):
    r"""Positive orthant :math:`\mathbb{R}_{>0}^d` via the elementwise log map.

    Forward map :math:`x = \exp(y)`, inverse :math:`y = \log x`. The Jacobian is
    diagonal with :math:`\partial x_i / \partial y_j = \exp(y_i)\,\delta_{ij}`, so

    .. math::
        \log|\det J| = \sum_i y_i.

    Hence ``latent_log_prob(y) = f(e^y) + \sum_i y_i`` and, by the chain rule,
    ``latent_score(y) = e^y \odot s(e^y) + 1``.
    """

    name = "positive_orthant"

    def to_constrained(self, y: Array) -> Array:
        return jnp.exp(y)

    def to_unconstrained(self, x: Array) -> Array:
        return jnp.log(x)

    def log_det_jacobian(self, y: Array) -> Array:
        return jnp.sum(y, axis=-1)

    def latent_log_prob(self, target_log_prob: LogProbFn, y: Array) -> Array:
        x = jnp.exp(y)
        return target_log_prob(x) + jnp.sum(y, axis=-1)

    def latent_score(self, target_score: ScoreFn, y: Array) -> Array:
        x = jnp.exp(y)
        return x * target_score(x) + 1.0


class Simplex(Support):
    r"""Probability simplex :math:`\Delta^{n-1}` via a softmax augmentation.

    A latent state :math:`y \in \mathbb{R}^n` is mapped bijectively onto
    :math:`(F, r) \in \Delta^{n-1} \times \mathbb{R}` by

    .. math::
        F = \mathrm{softmax}(y), \qquad r = \sum_i y_i,

    with inverse :math:`y_i = \log F_i + (r - \sum_j \log F_j)/n`. The radial
    coordinate :math:`r` carries the degree of freedom that softmax discards and
    is given an auxiliary 1-D prior :math:`g` (default: standard Gaussian),
    independent of :math:`F`.

    The Jacobian determinant of :math:`y \mapsto (F_{1:n-1}, r)` is
    :math:`n \prod_i F_i`, so

    .. math::
        \log|\det J| = \log n + \sum_i \log F_i = \log n + r - n\,\mathrm{lse}(y),

    where :math:`\mathrm{lse}` is ``logsumexp``. The lifted log-density is

    .. math::
        \log \pi(y) = \log p(F) + \log g(r) + \log n + r - n\,\mathrm{lse}(y),

    and its gradient (the score), via :math:`\partial F_i/\partial y_j =
    F_i(\delta_{ij} - F_j)`, is

    .. math::
        \nabla_y \log \pi = g'(r) + F \odot (s - \langle F, s\rangle) + 1 - n F,

    with :math:`s = \nabla_F \log p(F)`. The middle term is the softmax-Jacobian
    contribution that the original implementation dropped.
    """

    name = "simplex"

    def __init__(
        self,
        radial_log_prob_fn: LogProbFn = None,
        radial_score_fn: ScoreFn = None,
    ):
        self.radial_log_prob_fn = radial_log_prob_fn if radial_log_prob_fn is not None else self._gaussian_log_prob
        self.radial_score_fn = radial_score_fn if radial_score_fn is not None else self._gaussian_score

    @staticmethod
    def _gaussian_log_prob(x: Array, loc: float = 0.0, scale: float = 1.0) -> Array:
        return jstats.norm.logpdf(x, loc, scale)
    
    @staticmethod
    def _gaussian_score(x: Array, loc: float = 0.0, scale: float = 1.0) -> Array:
        return -(x - loc) / (scale ** 2)
    
    def to_constrained(self, y: Array) -> Array:
        """Project a latent state onto the simplex (drops the radial coordinate)."""
        return jax.nn.softmax(y, axis=-1)

    def to_unconstrained(self, x: Array, r: Array | float = 0.0) -> Array:
        r"""Reconstruct a latent ``y`` from a simplex point ``x`` and radius ``r``.

        The pre-image of a simplex point is a whole line in :math:`\mathbb{R}^n`
        (softmax is shift-invariant), parametrized by the radial coordinate
        :math:`r = \sum_i y_i`. Choosing ``r`` selects one point on that line;
        the map is only a bijection once ``r`` is pinned, which is why ``r`` is
        kept explicit rather than hidden.

        The default ``r = 0`` is the centred gauge :math:`\sum_i y_i = 0`. It is
        deliberately the mode of the default standard-Gaussian radial prior
        ``g``, so initializing chains from a simplex point lands at the most
        probable radius under the augmented target. If the radial prior is
        changed to one with non-zero mode, this default should track that mode.

        Precondition: ``x`` lies on the simplex (``x > 0`` and ``sum(x) == 1``).
        An unnormalized ``x`` is silently rescaled by softmax and will not invert.
        """
        n = x.shape[-1]
        log_x = jnp.log(x)
        c = (r - jnp.sum(log_x, axis=-1, keepdims=True)) / n
        return log_x + c

    def forward(self, y: Array) -> tuple[Array, Array]:
        """Full bijection ``y -> (F, r)`` used for diagnostics/tests."""
        return jax.nn.softmax(y, axis=-1), jnp.sum(y, axis=-1)

    def log_det_jacobian(self, y: Array) -> Array:
        n = y.shape[-1]
        r = jnp.sum(y, axis=-1)
        lse = logsumexp(y, axis=-1)
        return jnp.log(n) + r - n * lse

    def latent_log_prob(self, target_log_prob: LogProbFn, y: Array) -> Array:
        F = jax.nn.softmax(y, axis=-1)
        r = jnp.sum(y, axis=-1)
        return (
            target_log_prob(F)
            + self.radial_log_prob_fn(r)
            + self.log_det_jacobian(y)
        )

    def latent_score(self, target_score: ScoreFn, y: Array) -> Array:
        n = y.shape[-1]
        F = jax.nn.softmax(y, axis=-1)
        r = jnp.sum(y, axis=-1, keepdims=True)
        s = target_score(F)
        inner = jnp.sum(F * s, axis=-1, keepdims=True)
        return self.radial_score_fn(r) + F * (s - inner) + 1.0 - n * F

class LowRank(Support):
    """Log-low-rank parametrization of an HFPD-OT plan with balanced gauge constraints."""
    
    name="low_rank"

    def __init__(self, II: int, JJ: int, rank: int, cost: Array, epsilon: float,
                 support: str = "positive_orthant", ridge: float = 0.0):
        if rank < 2:
            raise ValueError(f"rank must be >= 2 to embed the dual potentials; got {rank}.")
        if rank > min(II, JJ):
            raise ValueError(f"rank {rank} exceeds min(II, JJ) = {min(II, JJ)}.")
        self.II = int(II)
        self.JJ = int(JJ)
        self.rank = int(rank)
        self.support = make_support(support)
        self.epsilon = float(epsilon)
        # Gauge-breaking ridge: add -ridge*||theta||^2 to the latent log-prob. It gives the
        # flat gauge directions curvature -- crucially the UNBOUNDED softmax-shift for the
        # simplex (softmax(ell + c*1) = softmax(ell)), which otherwise lets ||theta|| run to
        # infinity while pi stays frozen (acc~0.9, R-hat~4.5). See sampler-highdim-findings.
        # NOTE: this is the crude fix -- it penalizes ALL of theta, so it distorts the
        # pi-marginal. The principled, non-distorting fix is a prior on the shift Sum(ell)
        # only (mirroring the Simplex radial prior), or the Jeffreys volume term 1/2 log det G.
        self.ridge = float(ridge)
        self.log_K = -jnp.asarray(cost).reshape(II, JJ) / epsilon  # fixed -C/eps offset

        # Fully-free coordinates: U (II, r) then V (JJ, r), m = r(II+JJ). The r**2 GL(r)
        # gauge directions are flat for log p(pi(theta)) (assess convergence on pi, not theta);
        # gauge_fix is the balanced (U^T U = V^T V) canonicalizer used only at warm-start time.
        self._n_u = self.II * self.rank
        self.num_free = self._n_u + self.JJ * self.rank

    def unpack(self, theta: Array) -> tuple[Array, Array]:
        """``theta`` ``(..., m)`` -> ``U`` ``(..., II, r)``, ``V`` ``(..., JJ, r)``.

        Batch-aware over any leading axes: a single state ``(m,)`` or a batch ``(C, N, m)``.
        """
        lead = theta.shape[:-1]
        U = theta[..., : self._n_u].reshape(*lead, self.II, self.rank)
        V = theta[..., self._n_u :].reshape(*lead, self.JJ, self.rank)
        return U, V

    def pack(self, U: Array, V: Array) -> Array:
        """``U`` ``(..., II, r)``, ``V`` ``(..., JJ, r)`` -> ``theta`` ``(..., m)``."""
        lead = U.shape[:-2]
        return jnp.concatenate([U.reshape(*lead, -1), V.reshape(*lead, -1)], axis=-1)

    # -- reconstruction (outer linear layer; inner exp = PositiveOrthant, softmax = Simplex) ------

    def log_plan(self, U: Array, V: Array) -> Array:
        r"""``(..., II, r), (..., JJ, r)`` -> log-plan ``(..., II, JJ)`` = ``U V^T - C/eps``.

        ``log_K = -C/eps`` (shape ``(II, JJ)``) broadcasts against any leading axes.
        """
        return jnp.einsum("...ir,...jr->...ij", U, V) + self.log_K

    def to_constrained(self, theta: Array) -> Array:
        """``theta`` ``(..., m)`` -> plan ``pi`` ``(..., II*JJ)``.

        The ``(II, JJ)`` log-plan is flattened to a single ``II*JJ`` axis *before* the inner
        support map, so the simplex normalizes **globally** (total mass 1, not row-wise) and
        the whole map is batch-aware over any leading axes.
        """
        U, V = self.unpack(theta)
        y = self.log_plan(U, V)                              # (..., II, JJ)
        y = y.reshape(*y.shape[:-2], self.II * self.JJ)      # (..., II*JJ)
        return self.support.to_constrained(y)                # exp / global softmax on the last axis
    
    @staticmethod
    def _generate_skew_symmetric(r):
        z = jnp.random.randn(r, r)
        return LowRank._extract_skew_symmetric_part(z)
        
    @staticmethod
    def _generate_symmetric(r):
        z = jnp.random.randn(r, r)
        return LowRank._extract_symmetric_part(z)
    
    @staticmethod
    def _extract_symmetric_part(matrix):
        return 0.5 * (matrix + matrix.T)
    
    @staticmethod
    def _extract_skew_symmetric_part(matrix):
        return 0.5 * (matrix - matrix.T)
        
    def compute_bal_gauge_proposal_drift(self, theta, sigma_eq) -> BalancedGaugeDrift:
        """
        In the balanced gauge constraint, the energy is split equally between U and V and is diagonal.
        Returns the symmetric and skew symmetric parts of the proposal -- tightly locked into the manifold.
        Accepted or rejected by the MH correction step.
        """

        U, V = self.unpack(theta)
        score_func = self.latent_score_function(theta)
    
        # Compute the unrestricted gradients.
        G_U, G_V = self.unpack(score_func)

        # First project the gradients into the internal space: `U.T @ G_U` and `V.T @ G_V`, and then compute their skew symmetric part
        A_U = 0.5 * (U.T @ G_U - G_U.T @ U)
        A_V = 0.5 * (V.T @ G_V - G_V.T @ V)

        # Compute the symmetric part. Since S_U = S_V = S_shared, we satisfy the balanced gauge constraint.
        S_U = 0.5 * (U.T @ G_U + G_U.T @ U)
        S_V = 0.5 * (V.T @ G_V + G_V.T @ V)
        S_shared = 0.25 * (S_U + S_V) / sigma_eq

        return BalancedGaugeDrift(
            shared_symmetric_part=S_shared,
            assymetric_u=A_U,
            assymetric_v=A_V
        )

    def propose_bal_gauge_constrained_step(self, theta, epsilon) -> StateSummary:
        """
        Propose a new state in the balanced gauge constraint manifold.
        Returns the proposed state and the noise matrices A_noise_U, A_noise_V, S_noise that gave rise to the proposal.
        These are used in the MH correction step."""
        
        # Compute the symmetric and skew symmetric parts of the drift in the proposed state
        balanced_gauge_drift = self.compute_bal_gauge_proposal_drift(theta, sigma_eq=1.0)

        # Geometry-matched noise
        A_noise_U = LowRank._generate_skew_symmetric(self.rank)
        A_noise_V = LowRank._generate_skew_symmetric(self.rank)
        S_noise = LowRank._generate_symmetric(self.rank)

        # Total velocity matrices, confined now to the rigid tangent space of the balanced gauge.
        Omega_U = (epsilon**2 / 2.0) * (balanced_gauge_drift.assymetric_u + balanced_gauge_drift.shared_symmetric_part) + epsilon * (A_noise_U + S_noise)
        Omega_V = (epsilon**2 / 2.0) * (balanced_gauge_drift.assymetric_v + balanced_gauge_drift.shared_symmetric_part) + epsilon * (A_noise_V + S_noise)
        Omega = self.support.pack(Omega_U, Omega_V)

        # Retract onto the manifold using the exponential matrix operator
        U, V = self.unpack(theta)
        U_prop = U @ jax.scipy.linalg.expm(Omega_U)
        V_prop = V @ jax.scipy.linalg.expm(Omega_V)

        theta_prop = self.support.pack(U_prop, V_prop)

        return StateSummary(
            state=theta_prop,
            balanced_gauge_drift=balanced_gauge_drift,
            matching_noise=NoiseMatrices(A_noise_U=A_noise_U,
                                         A_noise_V=A_noise_V,
                                         S_noise=S_noise),
            velocities=Omega
        )
    
    def invert_retraction(self, proposed_theta, curr_theta: Array) -> Tuple[NoiseMatrices, Array]:
        """
        Invert the retraction step to compute the backward noise matrices that would have produced the current state from the proposed state.
        """
        balanced_gauge_drift_back = self.compute_bal_gauge_proposal_drift(proposed_theta, sigma_eq=1.0)
        
        # Invert the retraction via a matrix log.
        U_prop, V_prop = self.unpack(proposed_theta)
        U_original, V_original = self.unpack(curr_theta)
        omega_u_back = jax.scipy.linalg.logm(jnp.linalg.inv(U_prop) @ U_original)
        omega_v_back = jax.scipy.linalg.logm(jnp.linalg.inv(V_prop) @ V_original)
        omega_back = self.pack(omega_u_back, omega_v_back)

        # Extract the noise matrices from the velocities.
        S_noise_back = LowRank._extract_symmetric_part(omega_u_back) - balanced_gauge_drift_back.shared_symmetric_part
        A_noise_U_back = LowRank._extract_skew_symmetric_part(omega_u_back) - balanced_gauge_drift_back.assymetric_u
        A_noise_V_back = LowRank._extract_skew_symmetric_part(omega_v_back) - balanced_gauge_drift_back.assymetric_v

        return NoiseMatrices(A_noise_U=A_noise_U_back, A_noise_V=A_noise_V_back, S_noise=S_noise_back), omega_back
    
    def evaluate_noise_log_density(self, noise_matrices: NoiseMatrices) -> float:
        """
        Evaluates the log proposal density of the noise matrices under the standard Gaussian distribution."""
        pass

    def compute_det_exp_mat_jacobian(self, velocity: Array) -> float:
        """
        Compute the determinant of the Jacobian of the matrix exponential map at a given matrix \Omega.
        The Jacobian of the matrix exponential is governed by the operator: \frac{1-exp(-ad_{\Omega})}{ad_{\Omega}}
        where ad_{\Omega}(X) = [\Omega, X] is the adjoint operator or equivalently, the Lie bracket.
        The determinant of this operator can be computed using the eigenvalues of the adjoint operator.
        More precisely, the differences between those eigenvalues \((\mu_i - \mu_j)\) tell you exactly how much space is stretched.

        Args:
            velocity (Array): The velocity at which to compute the Jacobian determinant.
        Returns:
            float: The log determinant of the Jacobian of the matrix exponential at the given velocity.
        """
        # First, compute the eignevalues of the velocity
        eigenvalues = jnp.linalg.eigvals(velocity)
        # Because the velocity is not necessarily symmetric, ethe eignevalues may be complex. Take the abs:
        eigenvalues = jnp.abs(eigenvalues)

        # Compute the pairwise differences -- the eigenvalues of the adjoint operator.
        # We construct an rxr matrix where each entry (i, j) is the difference between the i-th and j-th eigenvalue.
        pairwise_differences = eigenvalues[:, None] - eigenvalues[None, :]

        # Stabilize computation
        abs_small = jnp.abs(pairwise_differences) < 1e-6
        abs_large = ~ abs_small

        eval_func_values = jnp.zeros_like(pairwise_differences, dtype=jnp.float64)

        large_diffs = pairwise_differences[abs_large]
        numerator = 1.0 - jnp.exp(-large_diffs)
        adj_eigenvalues = jnp.log(numerator / large_diffs)
        eval_func_values[abs_large] = adj_eigenvalues

        # For small differences -- identical or close eignevalues, we use a Taylor expansion to avoid numerical instability.
        small_diffs = pairwise_differences[abs_small]
        # taylor expansion of log(1 - exp(-x)) / x around x=0 is 1 - x/2 + x^2/6 - x^3/24 + ...
        taylor_expansion = 1 - small_diffs/2 + (small_diffs**2)/6 - (small_diffs**3)/24
        eval_func_values[abs_small] = taylor_expansion
        return jnp.sum(eval_func_values)


    def evaluate_bal_gauge_log_proposal_density(self, proposed_state: StateSummary,) -> float:
        """
        Evaluate the log proposal density of the proposed state given the noise matrices.
        This includes the log density of the noise matrices and the Jacobian determinant of the retraction step.
        """
        log_density = self.evaluate_noise_log_density(proposed_state.matching_noise) + self.compute_det_exp_mat_jacobian(proposed_state.velocities)
        return log_density
        
    def to_unconstrained(self, pi: Array) -> Array:
        r"""Warm start: a plan ``pi`` -> free vector via best rank-``r`` log-factorization.

        Factorizes ``L = log pi + C/eps`` by truncated SVD to rank ``r`` with the
        *balanced* split ``U = U_s sqrt(sigma)``, ``V = V_t sqrt(sigma)`` (symmetric
        magnitudes, no gauge fixing -> otherwise ``U_top^{-1}`` blow-up), then :meth:`pack`.
        For ``r=2`` and a Sinkhorn/EOT plan (``L`` exactly additively separable, hence
        rank 2) this round-trips the plan exactly -- the A bridge / correctness oracle.
        Single-plan (called once at warm-start), so it is *not* batched.
        """
        pi = jnp.asarray(pi)
        pi_mat = pi.reshape(*pi.shape[:-1], self.II, self.JJ)     # (..., II, JJ)
        L = jnp.log(jnp.clip(pi_mat, 1e-300, None)) - self.log_K  # = log pi + C/eps, (..., II, JJ)
        Us, s, Vt = jnp.linalg.svd(L, full_matrices=False)        # (..., II, K), (..., K), (..., K, JJ)
        sr = jnp.sqrt(s[..., : self.rank])[..., None, :]          # (..., 1, r) -> broadcasts over rows
        U = Us[..., :, : self.rank] * sr                          # (..., II, r): first r columns of Us
        V = jnp.swapaxes(Vt[..., : self.rank, :], -1, -2) * sr    # (..., JJ, r): first r rows of Vt, transposed
        return self.pack(U, V)
    
    def latent_log_prob(self, target_log_prob_fn: LogProbFn, theta: Array) -> Array:
        # NOTE: v1 omits the Jeffreys volume term 1/2 log det G(theta); this targets the
        # pullback log p(pi(theta)) plus the gauge-breaking ridge. Per-state (vmapped).
        pi = self.to_constrained(theta).reshape(1, self.II * self.JJ)
        return target_log_prob_fn(pi) - self.ridge * jnp.sum(theta ** 2)

    def latent_score(self, target_score_fn: ScoreFn, theta: Array, with_ridge: bool = False) -> Array:
        """Exact chain rule of ``log p(pi(theta))`` through ``pi = to_constrained(U V^T - C/eps)``.

        Called per-state (the sampler vmaps over chains); ``target_score_fn`` is single-state,
        so this is not vectorized over a ``(C, N)`` batch, but the einsums are ``...``-shaped.
        """
        orig_shape = theta.shape
        U, V = self.unpack(theta)                                       # (..., II, r), (..., JJ, r)
        pi = self.to_constrained(theta)                                # (..., II*JJ)
        pi_mat = pi.reshape(*pi.shape[:-1], self.II, self.JJ)          # (..., II, JJ)
        s_pi = target_score_fn(pi.reshape(1, self.II * self.JJ)).reshape(self.II, self.JJ)
        M = s_pi * pi_mat                                             # orthant: pi ⊙ s
        
        if self.support.name == "simplex":                           # softmax adds the centering
            M = M - pi_mat * jnp.sum(M, axis=(-2, -1), keepdims=True)  # pi ⊙ (s - <pi, s>)
        grad_U = jnp.einsum("...ij,...jr->...ir", M, V)              # d log p / d U = M V
        grad_V = jnp.einsum("...ij,...ir->...jr", M, U)              # d log p / d V = M^T U
        grad = self.pack(grad_U, grad_V).reshape(orig_shape)
        
        if with_ridge:
            grad = grad - 2.0 * self.ridge * theta                       # + gradient of -ridge*||theta||^2
        return grad


class LowRankSection(LowRank):
    r"""Option B: hard-gauge flat *section* of the log-low-rank manifold ``M_Phi``.

    Samples the ``m = r(II+JJ) - r**2`` free coordinates ``theta = (U_bot, V)`` with the top
    ``r x r`` block of ``U`` **pinned to** ``I_r`` -- a *complete, FLAT* gauge fix (§2.4 hard
    gauge). Because the chart is flat, ordinary additive MALA needs no retraction and moves
    **both** the core and the subspace (perturbing ``U_bot`` tilts ``col(U)``; see §3). The
    target is the Hausdorff measure on ``M_Phi``, i.e. the pullback

        log q(theta) = log p(pi(theta)) + 1/2 log det G_S(theta),
        G_S = J^T J,   J = d pi / d theta   (shape ``(II*JJ, m)``),

    with **no ridge** -- the gauge is fixed by the chart, not softly broken. ``1/2 log det G_S``
    is evaluated by exact autodiff of the section map (jacobian -> ``slogdet``); matrix-free is
    the §4 TODO. This is a deliberately asymmetric *diagnostic* chart (the balanced Stiefel
    version is Option C); its ``name`` is not ``"low_rank"`` so the sampler routes it through
    the generic symmetric-proposal MALA path.
    """

    name = "low_rank_section"

    def __init__(self, II: int, JJ: int, rank: int, cost: Array, epsilon: float,
                 support: str = "positive_orthant", **_ignored):
        # ridge=0: the flat section fixes the gauge, so no gauge-breaking penalty is needed.
        super().__init__(II=II, JJ=JJ, rank=rank, cost=cost, epsilon=epsilon,
                         support=support, ridge=0.0)
        self._n_ubot = (self.II - self.rank) * self.rank
        self.num_free = self._n_ubot + self.JJ * self.rank      # m = r(II+JJ) - r**2
        # Pivoted hard gauge: pin the best-conditioned r rows to I_r (a fixed top block is
        # ill-conditioned -- inv(U_top) blows the warm start up by ~100x, freezing the chain).
        # Identity until to_unconstrained picks the pivot from the warm-start plan, then fixed
        # for the chart's lifetime (baked into the jit trace, which happens after warm start).
        self.perm = jnp.arange(self.II)
        self.inv_perm = jnp.arange(self.II)

    # -- section (un)packing: theta = (U_bot, V); U reconstructed with a pinned I_r block ------

    def unpack(self, theta: Array) -> tuple[Array, Array]:
        """``theta`` ``(..., m)`` -> full ``U`` ``(..., II, r)`` (top block ``I_r``), ``V`` ``(..., JJ, r)``."""
        lead = theta.shape[:-1]
        U_bot = theta[..., : self._n_ubot].reshape(*lead, self.II - self.rank, self.rank)
        V = theta[..., self._n_ubot :].reshape(*lead, self.JJ, self.rank)
        I_r = jnp.broadcast_to(jnp.eye(self.rank, dtype=theta.dtype), (*lead, self.rank, self.rank))
        U_tilde = jnp.concatenate([I_r, U_bot], axis=-2)        # permuted-space U (top block I_r)
        U = jnp.take(U_tilde, self.inv_perm, axis=-2)           # restore physical row order
        return U, V

    def pack_section(self, U_bot: Array, V: Array) -> Array:
        """``U_bot`` ``(..., II-r, r)``, ``V`` ``(..., JJ, r)`` -> ``theta`` ``(..., m)``."""
        lead = U_bot.shape[:-2]
        return jnp.concatenate([U_bot.reshape(*lead, -1), V.reshape(*lead, -1)], axis=-1)

    # -- volume term 1/2 log det G_S (exact-autodiff embedding Jacobian) -----------------------

    def _volume_term(self, theta_flat: Array) -> Array:
        """``1/2 log det(J^T J)``, ``J = d pi/d theta`` on the section. ``theta_flat`` is ``(m,)``.

        A small jitter ``+ eps I`` bounds ``inv(G)`` at chart-edge degeneracies (where ``G``
        loses rank), so the value and its autodiff gradient stay finite -- otherwise a chain
        that wanders near a fold gets ``nan`` gradients and freezes.
        """
        J = jax.jacobian(self.to_constrained)(theta_flat)      # (II*JJ, m)
        G = J.T @ J                                            # (m, m), full-rank on the section
        G = G + 1e-6 * jnp.eye(G.shape[0], dtype=G.dtype)
        return 0.5 * jnp.linalg.slogdet(G)[1]

    def latent_log_prob(self, target_log_prob_fn: LogProbFn, theta: Array) -> Array:
        pi = self.to_constrained(theta).reshape(1, self.II * self.JJ)
        return target_log_prob_fn(pi) + self._volume_term(theta.reshape(-1))

    def latent_score(self, target_score_fn: ScoreFn, theta: Array) -> Array:
        """Analytic chain-rule score of ``log p(pi(theta))`` (restricted to the free coords)
        plus the exact-autodiff gradient of the volume term."""
        t = theta.reshape(-1)
        U, V = self.unpack(t)                                  # full U (II, r), V (JJ, r)
        pi = self.to_constrained(t)
        pi_mat = pi.reshape(self.II, self.JJ)
        s_pi = target_score_fn(pi.reshape(1, self.II * self.JJ)).reshape(self.II, self.JJ)
        M = s_pi * pi_mat                                      # orthant: pi (.) s
        if self.support.name == "simplex":                    # softmax centering
            M = M - pi_mat * jnp.sum(M)
        grad_U = M @ V                                         # (II, r), physical row order
        grad_V = M.T @ U                                       # (JJ, r)
        # Free coords are the permuted U_bot (top r rows pinned to I_r), so permute the row
        # gradient before dropping the pinned top block.
        score_logp = self.pack_section(grad_U[self.perm][self.rank :, :], grad_V)   # (m,)
        vol_grad = jax.grad(self._volume_term)(t)             # (m,)
        return (score_logp + vol_grad).reshape(theta.shape)

    # -- warm start: raw balanced (U,V) -> section coords (project U_top -> I_r) ---------------

    def to_unconstrained(self, pi: Array) -> Array:
        r"""Warm start: plan ``pi`` -> section coords. Best rank-``r`` log-factorization, then
        the gauge move ``R = U_top^{-1}`` sends ``U_top -> I_r`` (``V -> V U_top^T`` keeps
        ``U V^T``). Single-plan (called once), not batched."""
        pi = jnp.asarray(pi).reshape(-1)
        L = jnp.log(jnp.clip(pi.reshape(self.II, self.JJ), 1e-300, None)) - self.log_K
        Us, s, Vt = jnp.linalg.svd(L, full_matrices=False)
        sr = jnp.sqrt(s[: self.rank])[None, :]                 # (1, r)
        U = Us[:, : self.rank] * sr                            # (II, r)
        V = (Vt[: self.rank, :].T) * sr                        # (JJ, r)
        # Pick the best-conditioned r rows for the I_r block via LU partial pivoting (row perm).
        P, _, _ = jax.scipy.linalg.lu(U)
        self.perm = jnp.argmax(P, axis=0)                      # physical -> permuted row order
        self.inv_perm = jnp.argsort(self.perm)
        U_t = U[self.perm]                                     # top r rows now well-conditioned
        U_top = U_t[: self.rank, :]                            # (r, r)
        U_sec = U_t @ jnp.linalg.inv(U_top)                    # top block -> I_r
        V_sec = V @ U_top.T                                    # preserves U V^T = U_sec V_sec^T
        return self.pack_section(U_sec[self.rank :, :], V_sec)


class WhitenedSupport(Support):
    r"""Wrap a support ``inner`` with a fixed linear preconditioner ``L`` (``m x m``).

    Samples are drawn in **whitened** coordinates ``phi`` with ``theta = L phi``; an *isotropic*
    proposal on ``phi`` therefore has dense metric ``M = L L^T`` on ``theta``. Choosing ``L`` as a
    SoftAbs--Laplace whitener at the mode (``M ~ [-d^2 log q(theta*)]^{-1}``, :meth:`laplace`)
    turns a stiff, *correlated* target into a well-conditioned (kappa ~ 1) one -- which a diagonal
    mass cannot do (it cannot un-tilt correlated level sets). ``|det L|`` is constant, so it
    cancels in the MH ratio and every downstream method just composes with ``L``. ``name`` and
    ``num_free`` delegate to ``inner`` so the sampler treats this as the same support.
    """

    def __init__(self, inner: Support, L: Array, L_inv: Array):
        self.inner = inner
        self.L = L
        self.L_inv = L_inv
        self.name = inner.name
        self.num_free = inner.num_free

    @classmethod
    def from_mode_hessian(cls, inner: Support, H: Array, floor: float = 1e-3) -> "WhitenedSupport":
        """Build a **constant** whitener from the mode Hessian ``H = -d^2 log q(theta*)``.

        The resulting ``M = [H]^{-1}`` (regularized) equals the *covariance of the Laplace
        approximation*.

        Metric eigenvalues ``max(|lambda|, floor)`` -- absolute value handles indefinite/saddle
        directions, the floor bounds the step in near-flat directions. ``L = Q diag(1/sqrt(d))``
        gives ``M = L L^T = Q diag(1/d) Q^T``.

        NOTE: this is *not* Betancourt's SoftAbs metric. It borrows only the abs-eigenvalue idea:
        (i) it is evaluated **once at the mode** (a fixed preconditioner), not position-dependent;
        (ii) it uses a hard ``max(|lambda|, floor)`` rather than the smooth ``lambda*coth(alpha*lambda)``.
        The hard/smooth distinction is immaterial for a constant metric (never differentiated); the
        constant-vs-position-dependent one is the real gap -- the position-dependent metric is a
        separate experiment, and for this manifold the natural choice there is the induced metric
        ``G(theta) = J^T J`` (PSD by construction, already computed for the volume term), not softabs(H).
        """
        w, Q = jnp.linalg.eigh(0.5 * (H + H.T))               # symmetrize for numerical safety
        d = jnp.maximum(jnp.abs(w), floor)                    # abs eigenvalues, floored (not softabs)
        L = Q * (1.0 / jnp.sqrt(d))                           # (m,m): Q @ diag(1/sqrt d)
        L_inv = jnp.sqrt(d)[:, None] * Q.T                    # diag(sqrt d) @ Q^T
        return cls(inner, L, L_inv)

    def _theta(self, phi: Array) -> Array:
        return phi @ self.L.T                                 # theta = L phi (batched)

    def to_constrained(self, phi: Array) -> Array:
        return self.inner.to_constrained(self._theta(phi))

    def to_unconstrained(self, pi: Array) -> Array:
        return self.inner.to_unconstrained(pi) @ self.L_inv.T  # phi = L^{-1} theta

    def latent_log_prob(self, target_log_prob_fn: LogProbFn, phi: Array) -> Array:
        return self.inner.latent_log_prob(target_log_prob_fn, self._theta(phi))

    def latent_score(self, target_score_fn: ScoreFn, phi: Array) -> Array:
        g = self.inner.latent_score(target_score_fn, self._theta(phi))  # d/dtheta
        return g @ self.L                                     # d/dphi = L^T g


def make_support(
    kind: str,
    sampling_strategy: Literal["low_rank", "low_rank_section", "full_rank"] = "full_rank",
    *,
    radial_log_prob_fn: LogProbFn | None = None,
    radial_score_fn: ScoreFn | None = None,
    **kwargs
) -> Support:
    """Factory: ``"unconstrained"`` | ``"simplex"`` | ``"positive_orthant"`` | ``"low_rank"``.

    The radial priors parametrize the simplex augmentation only; they are
    ignored for the unconstrained and positive-orthant supports.
    """
    if sampling_strategy == "full_rank":
        if kind == "simplex":
            return Simplex(radial_log_prob_fn=radial_log_prob_fn, radial_score_fn=radial_score_fn)
        if kind == "unconstrained":
            return Unconstrained()
        if kind == "positive_orthant":
            return PositiveOrthant()
        raise ValueError(
            f"Unknown support {kind!r}; expected 'unconstrained', 'simplex' or 'positive_orthant'."
        )

    elif sampling_strategy == "low_rank":
        return LowRank(
            II=kwargs.get("II", 1),
            JJ=kwargs.get("JJ", 1),
            rank=kwargs.get("rank", 2),
            cost=kwargs.get("cost", jnp.zeros((1, 1))),
            epsilon=kwargs.get("epsilon", 1.0),
            support=kind,  # a STRING ("positive_orthant"/"simplex"); LowRank.__init__ wraps it via make_support
            ridge=kwargs.get("ridge", 0.0) or 0.0,
        )

    elif sampling_strategy == "low_rank_section":
        # Option B: hard-gauge flat section + 1/2 log det G_S volume term (no ridge).
        return LowRankSection(
            II=kwargs.get("II", 1),
            JJ=kwargs.get("JJ", 1),
            rank=kwargs.get("rank", 2),
            cost=kwargs.get("cost", jnp.zeros((1, 1))),
            epsilon=kwargs.get("epsilon", 1.0),
            support=kind,
        )
    else:
        raise ValueError(
            f"Unknown support {kind!r}; expected 'unconstrained', 'simplex', 'positive_orthant' or 'low_rank'."
        )