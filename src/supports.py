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
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp
import jax.scipy.stats as jstats

LogProbFn = Callable[[Array], Array]
ScoreFn = Callable[[Array], Array]


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

    # -- gauge handling and warm start -----------------------------------------

    def gauge_fix(self, U: Array, V: Array) -> tuple[Array, Array]:
        """Unlike the top-block chart (``R = U_top^{-1}``, which blows up when the top
        ``r x r`` block is ill-conditioned), this uses **no inverse**: QR both factors,
        SVD the ``r x r`` core, and redistribute ``sqrt`` of the singular values so that
        ``U'^T U' = V'^T V' = diag(sigma)`` while ``U' V'^T = U V^T`` is unchanged. Cost
        ``O((II+JJ) r^2 + r^3)``. The SVD warm start is already balanced, so on it this
        map is ~identity (no blow-up)."""

        Qu, Tu = jnp.linalg.qr(U)                  # U = Qu Tu,  Qu:(II,r)  Tu:(r,r)
        Qv, Tv = jnp.linalg.qr(V)                  # V = Qv Tv,  Qv:(JJ,r)  Tv:(r,r)
        W, sigma, Zt = jnp.linalg.svd(Tu @ Tv.T)   # r x r core SVD
        root = jnp.sqrt(sigma)
        U_bal = Qu @ (W * root)                # columns scaled by sqrt(sigma_k)
        V_bal = Qv @ (Zt.T * root)
        return U_bal, V_bal

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

    def latent_score(self, target_score_fn: ScoreFn, theta: Array) -> Array:
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
        return grad - 2.0 * self.ridge * theta                       # + gradient of -ridge*||theta||^2


def make_support(
    kind: str,
    sampling_strategy: Literal["low_rank", "full_rank"] = "full_rank",
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
    else:
        raise ValueError(
            f"Unknown support {kind!r}; expected 'unconstrained', 'simplex', 'positive_orthant' or 'low_rank'."
        )