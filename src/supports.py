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
from typing import Callable

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


def make_support(
    kind: str,
    *,
    radial_log_prob_fn: LogProbFn | None = None,
    radial_score_fn: ScoreFn | None = None,
) -> Support:
    """Factory: ``"unconstrained"`` | ``"simplex"`` | ``"positive_orthant"``.

    The radial priors parametrize the simplex augmentation only; they are
    ignored for the unconstrained and positive-orthant supports.
    """
    if kind == "simplex":
        return Simplex(radial_log_prob_fn=radial_log_prob_fn, radial_score_fn=radial_score_fn)
    if kind == "unconstrained":
        return Unconstrained()
    if kind == "positive_orthant":
        return PositiveOrthant()
    raise ValueError(
        f"Unknown support {kind!r}; expected 'unconstrained', 'simplex' or 'positive_orthant'."
    )
