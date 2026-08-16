from dataclasses import dataclass
from functools import partial
import logging
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from datetime import date
from jax import Array
import jax.numpy as jnp
from jax.experimental import mesh_utils
from typing import Dict, Sequence, Tuple, Literal
from supports import make_support, WhitenedSupport, LogProbFn, ScoreFn
from mcmc_diagnostics import DiagnosticsSummary, MCMCDiagnostics

# Set up the XLA flag to use jax.pmap on CPU
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={multiprocessing.cpu_count()}"

devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
MESH = Mesh(devices, axis_names=("chains", ))
SHARDING = NamedSharding(MESH, P("chains"))

# NB: do not force x64 on/off at import — that leaks into any importing process
# (e.g. the test suite). Precision is a runtime choice of the caller.

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info(f"Number of devices: {jax.local_devices()}")
TARGET_ACCEPT = 0.574

# Registered as a pytree so it can be the carry of jax.lax.scan / vmap (all fields are
# dynamic arrays -> data_fields; no static metadata).
@partial(jax.tree_util.register_dataclass,
         data_fields=["state", "counter", "rng_keys", "step_size",
                      "num_accepted_samples", "avg_mh_ratio", "latent_log_prob_state"],
         meta_fields=[])
@dataclass
class MALAState:
    """Holds the intermediate state of the MALA sampler while running `inference_loop`.

    ``latent_log_prob_state`` is the log-density of the CURRENT (accepted) latent state --
    carried through the scan so eBFMI can read it directly at diagnostics time instead of
    re-evaluating log_prob on every draw (which materialises a full plan per draw -> OOM at
    cell scale). All fields are dynamic arrays (data_fields); registration order MUST match
    the field list below.
    """
    state: Array
    counter: Array
    rng_keys: Array
    step_size: Array
    num_accepted_samples: Array
    avg_mh_ratio: Array
    latent_log_prob_state: Array | None = None

# Registered as a pytree so vmap/jit can return it (one instance whose fields carry a leading
# chain axis -- NOT a list of per-chain instances). Access fields directly, e.g.
# `stacked.stacked_latent_states` has shape (C, num_steps, ...); do NOT index `stacked[i]`.
@partial(jax.tree_util.register_dataclass,
         data_fields=["stacked_latent_states", "stacked_latent_log_prob_states",
                      "stacked_num_accepted_samples", "stacked_avg_mh_ratios"],
         meta_fields=[])
@dataclass
class StackedMALAStates:
    r"""
    Holds the stacked states of the MALA sampler after running `inference_loop`, for all steps and one chain.
    """
    stacked_latent_states: Array
    stacked_latent_log_prob_states: Array
    stacked_num_accepted_samples: Array
    stacked_avg_mh_ratios: Array


@dataclass
class MalaChainSummary:
    r"""Holds the summary states -- aggregated over all steps -- of the MALA sampler after running one chain of `inference_loop`.

    ``latent_samples`` (the compact ``theta``, ~250x smaller than a plan) is the **manifest**: any
    plan is rebuildable from it via ``support.to_constrained``. The full set of plans is not
    storable at cell scale (10^6 plan x 5_000 draws x 4 chains = 80 GB > 52 GB).

    ``samples`` are the plans ``pi`` -- but with ``sample(max_pi_coords=k)`` they are **per-parameter
    thinned**: only a fixed random ``k``-coordinate subset is kept (``pi_coord_idx`` records which).
    That subset is chosen ONCE and applied to every draw, so each retained coordinate keeps its
    complete draw-trace and its R-hat/ESS are **exact and uncapped**.

    NOTE: a per-parameter-thinned ``samples`` is for **diagnostics only** -- it cannot be
    push-forwarded (``psi . pi_bar`` needs the whole ``II x JJ`` matrix). Downstream consumers must
    rebuild the ``N`` whole plans they need from ``latent_samples`` and stream them.
    """
    samples: Array
    diagnostics: DiagnosticsSummary | None = None
    pi_coord_idx: Array | None = None     # set iff `samples` is a trimmed coordinate subset
    config: Dict | None = None
    stacked_mala_states: StackedMALAStates | None = None


class MetropolisAdjustedLangevinSampler:
    """
    Implements the Metropolis-Adjusted Langevin Algorithm (MALA) transition kernel for MCMC sampling in very high-dimensional
    spaces, with optional support for sampling from distributions constrained to the simplex or positive orthant. The kernel uses a preconditioned
    Langevin update step, followed by a Metropolis-Hastings correction step to ensure detailed balance and convergence
    to the target distribution.
    """

    def __init__(
        self,
        target_log_prob_fn: LogProbFn,
        target_score_fn: ScoreFn,
        radial_log_prob_fn: LogProbFn | None = None,
        radial_score_fn: ScoreFn | None = None,
        alpha: float| None = 0.5,
        shape: int = 10,
        sampling_strategy: Literal["low_rank", "low_rank_section", "full_rank"] = "full_rank",
        support: Literal["simplex", "positive_orthant", "unconstrained"] = "unconstrained",
        num_samples: int = 8_000,
        num_burnin: int = 8_000,
        warm_up_steps: int = 500,
        step_size: float = 0.1,
        num_parallel_chains: int = jax.local_device_count(),
        seed: int = 0,
        initial_plan: Array | None = None,
        warm_start_sigma: tuple[float, float] = (0.1, 0.3),
        precondition: Literal["mode_hessian"] | None = None,
        precond_floor: float = 1e-3,
        warmup: Literal["windowed", "legacy"] = "windowed",
        **kwargs,
    ):
        """Initializes the MetropolisAdjustedLangevinSampler transition kernel.

        Args:
           target_log_prob_fn: Python callable which takes the current state as argument and returns its (unnormalised)
           log-density under the target distribution.
           target_log_prob_grad: Python callable which takes the current state as argument and returns
           the score function of the target distribution evaluated at this state.
           alpha_0, beta_0 and gamma_0 (int): parameters controlling the adaptive learning rate.
           parallel_chains: number of chains.

        Returns:
            next_state: Python list of NDArray representing the state(s) of the Markov chain(s) at each resulting step. Has same shape as
            `current_state`.
            kernel_results: `collections.namedtuple` of internal calculations used to advance the chain.
        """
        self.target_log_prob_function = target_log_prob_fn
        self.target_score_function = target_score_fn
        self.shape = shape
        self.parallel_chains = num_parallel_chains
        self.num_burnin = num_burnin
        self.warm_up_steps = warm_up_steps
        # Warm-up scheme: "windowed" = continuous step adaptation toward TARGET_ACCEPT with
        # expanding-window within-chain mass estimation, kernel FROZEN for the retained
        # draws; "legacy" = the historical flow (unadapted warm-up, pooled-variance mass,
        # adaptation through the retained draws) kept only for E-series reproduction.
        self.warmup = warmup
        self.step_size = step_size
        self.seed = seed
        # Optional constrained-space starting point (e.g. an EOT/UOT Sinkhorn plan) to seed
        # chains near the mode; problem-specific, supplied by the caller (separation of concerns).
        self.initial_plan = initial_plan
        # Per-chain warm-start perturbation scale (lo..hi across chains); over-dispersion
        # relative to the target is what makes R-hat a meaningful convergence signal.
        self.warm_start_sigma = warm_start_sigma
        self.init_key = jax.random.key(self.seed)
        self.alpha = alpha ## smoothing factor of the exponential moving average
        self.num_samples = num_samples
        self.tot_num_samples = self.num_samples + self.num_burnin
        self.rank = kwargs.get("rank", None)

        # If support low_rank, make the II, JJ, rank and cost required
        if sampling_strategy in ("low_rank", "low_rank_section"):
            if "II" not in kwargs or "JJ" not in kwargs or "rank" not in kwargs or "cost" not in kwargs:
                raise ValueError("For low_rank support, II, JJ, rank and cost must be provided as keyword arguments.")
            if kwargs["II"] <= 0 or kwargs["JJ"] <= 0 or kwargs["rank"] <= 0:
                raise ValueError("II, JJ and rank must be positive integers.")
            if kwargs["rank"] > min(kwargs["II"], kwargs["JJ"]):
                raise ValueError("rank must be less than or equal to min(II, JJ).")
            if not isinstance(kwargs["cost"], Array):
                raise ValueError("cost must be a jax.numpy Array.")

        self.support = make_support(
            support,
            sampling_strategy,
            radial_log_prob_fn=radial_log_prob_fn,
            radial_score_fn=radial_score_fn,
            II=kwargs.get("II", None),
            JJ=kwargs.get("JJ", None),
            rank=self.rank,
            cost=kwargs.get("cost", None),
            epsilon=kwargs.get("epsilon", None),
            ridge=kwargs.get("ridge", 0.0),
        )

        # Low-rank supports define their own free-coordinate count m (r(II+JJ), or
        # r(II+JJ)-r**2 for the hard-gauge section); use it so the caller need not pass it.
        if hasattr(self.support, "num_free"):
            self.shape = self.support.num_free

        # Optional ridge centering: "warm_start" places the ridge at theta_0 = the chart
        # coordinates of initial_plan (exact rank-2 factorisation of the certainty-equivalent
        # plan pi^o when initial_plan = sinkhorn_init). Uncentered (None, the default) penalises
        # ||theta||^2, whose zero maps to the ideal design pi_I -- a first-order ideal-ward pull
        # on every observable. An explicit array is accepted for custom centres.
        ridge_center = kwargs.get("ridge_center", None)
        if ridge_center is not None and hasattr(self.support, "ridge_center"):
            if isinstance(ridge_center, str):
                if ridge_center != "warm_start":
                    raise ValueError(f"ridge_center must be 'warm_start', an array, or None; got {ridge_center!r}")
                if self.initial_plan is None:
                    raise ValueError("ridge_center='warm_start' requires an initial_plan.")
                ridge_center = self.support.to_unconstrained(
                    self.initial_plan.reshape(1, -1)).reshape(self.shape)
            self.support.ridge_center = jnp.asarray(ridge_center).reshape(self.shape)

        # Optional constant dense preconditioner: whiten by the mode-Hessian metric,
        # M = [-d^2 log q(theta*)]^{-1} (experiment 1). Wraps the support so the sampler runs
        # isotropically in whitened coords; needs a warm-start mode to evaluate the Hessian.
        if precondition == "mode_hessian":
            if self.initial_plan is None:
                raise ValueError("precondition='mode_hessian' requires an initial_plan (the mode).")
            inner = self.support
            theta_star = inner.to_unconstrained(self.initial_plan.reshape(1, -1)).reshape(self.shape)
            neg_logq = lambda t: -jnp.sum(inner.latent_log_prob(self.target_log_prob_function, t))
            H = jax.hessian(neg_logq)(theta_star)
            self.support = WhitenedSupport.from_mode_hessian(inner, H, floor=precond_floor)

        self.latent_log_prob_function = lambda x: self.support.latent_log_prob(self.target_log_prob_function, x)
        self.latent_score_function = lambda x: self.support.latent_score(self.target_score_function, x)

    def sample(
        self,
        with_diagnostics: bool = True,
        max_pi_coords: int | None = None,):
        r"""Run the chains and summarise them.

        Args:
            with_diagnostics: compute the R-hat / ESS / eBFMI summary.
            max_pi_coords: **per-parameter thinning** -- keep only this many coordinates of ``pi``.
                Leave ``None`` for small plans (full behaviour). Needed at cell scale: the chain
                itself is compact (``theta``), but expanding every draw into a whole plan at once
                costs ``chains x draws x II*JJ`` -- 80 GB for a 10^6 plan, 4 chains, 5000 draws.
        """
        # Sample all chains. `mix` is a SINGLE vmapped StackedMALAStates whose fields carry a
        # leading chain axis -- access fields directly.
        mix = self.inference_loop_multiple_chains()

        latent_samples = mix.stacked_latent_states                 # (C, tot_steps, 1, m)
        latent_log_prob_states = mix.stacked_latent_log_prob_states  # (C, tot_steps, 1, 1)

        # Drop burn-in + singleton axes. The energies are carried per step (the accepted state's
        # log-prob, ALREADY computed for the MH ratio), so eBFMI reads them with ZERO extra compute
        # and exactly for the states visited. NB: recomputing them is not inherently OOM -- each
        # energy needs only one transient plan -- but the OLD diag path recomputed via vmap over
        # ALL draws, materialising the (C, D, II*JJ) intermediate at once = 80 GB. The carry
        # sidesteps both the recompute and that batching.
        latent_samples = jnp.squeeze(latent_samples[:, self.num_burnin:], axis=2)   # (C, D, m)
        C, D = latent_samples.shape[0], latent_samples.shape[1]
        latent_log_prob_states = latent_log_prob_states[:, self.num_burnin:].reshape(C, D)  # (C, D)

        pi_coord_idx = None
        plan_dim = self.support.to_constrained(latent_samples[0, 0]).shape[-1]   # one plan only
        if max_pi_coords is not None and plan_dim > max_pi_coords:
            # PER-PARAMETER THINNING. The coordinate subset is drawn ONCE and closed over, so the
            # SAME coordinates are kept in every draw. That fixedness is what makes it sound: each
            # retained coordinate keeps its complete draw-trace, so its autocorrelation -- hence
            # ESS_j and R-hat_j -- is EXACT and uncapped. (A fresh subset per draw would give
            # ragged traces and meaningless ESS.) R-hat/ESS are per-coordinate statistics reported
            # as min/median, so coordinates are interchangeable replicates.
            #
            # TODO(revisit): this diagnoses a SUBSET of the parameter space. If an EXHAUSTIVE
            # per-parameter diagnostic (all II*JJ coordinates) is ever needed, switch axes: thin
            # the DRAWS instead and keep whole plans (as Patterson & Teh, NIPS 2013, do with
            # thin=100), accepting that ESS is then capped by the retained draw count.
            pi_coord_idx = jax.random.choice(jax.random.key(self.seed + 7777), plan_dim,
                                             shape=(max_pi_coords,), replace=False)
            C, D, m = latent_samples.shape
            # Flatten to (C*D, m) so lax.map iterates over INDIVIDUAL draws: only one whole plan
            # (~4 MB) is live at a time, never the (C, D, II*JJ) block. It also makes the lambda
            # receive a single state (m,), so `[pi_coord_idx]` indexes the COORDINATE axis -- on an
            # unflattened (D, n) slice it would silently select draws instead.
            constrained_samples = jax.lax.map(
                lambda th: self.support.to_constrained(th)[pi_coord_idx],
                latent_samples.reshape(-1, m),
            ).reshape(C, D, max_pi_coords)
        else:
            constrained_samples = self.support.to_constrained(latent_samples)   # (C, D, II*JJ)

        # R-hat/ESS are computed on the CONSTRAINED (pi) coordinates: on the latent chart they
        # would be polluted by gauge drift (theta can move along GL(r) orbits without changing
        # pi), whereas on pi they are gauge-invariant. eBFMI is the exception -- it needs the
        # energy, so it reads the latent log-prob trace.
        # The chains are pulled to host FIRST: `device_get` on a sharded array is a per-shard D2H
        # copy (no collective), whereas reducing over the sharded ("chains",) axis on device issues
        # an NCCL all-to-all inside `jit__reduce_max`, which fails on hosts where that collective is
        # unavailable. On host input the reductions run on the default device only. Diagnostics are
        # O(C*N*D) once per run, so the transfer cost is negligible next to sampling.
        diag = None
        if with_diagnostics:
            self._diagnostics = MCMCDiagnostics(
                constrained_chains=jax.device_get(constrained_samples),
                latent_log_prob_states=jax.device_get(latent_log_prob_states),
            )
            diag = self._diagnostics.summarize()

        return MalaChainSummary(
            samples=constrained_samples,
            diagnostics=diag,
            pi_coord_idx=pi_coord_idx,
            stacked_mala_states=mix,   # full archive (incl. burn-in): latent theta + acceptance counts
        )

    def thinned_plans(self, result, per_chain: int = 250, chunk: int = 100):
        """Retained draws as full constrained plans, thinned per chain -- host-side (n_kept, dim).

        The complement of ``max_pi_coords``: that caps which COORDINATES the diagnostics see,
        this caps which DRAWS are expanded into whole plans. Latent ``theta`` is the manifest
        (see ``MalaChainSummary``); materialising every retained draw as a plan costs
        ``chains x draws x II*JJ`` and is not feasible at production plan sizes, so the draws
        are thinned evenly per chain IN LATENT SPACE and mapped ``theta -> pi`` in chunks.
        """
        lat = np.asarray(jax.device_get(result.stacked_mala_states.stacked_latent_states))
        lat = lat.reshape(lat.shape[0], lat.shape[1], -1)[:, -self.num_samples:, :]
        keep = np.linspace(0, self.num_samples - 1,
                           min(per_chain, self.num_samples)).astype(int)
        lat = lat[:, keep, :].reshape(-1, lat.shape[-1])
        plans = None
        for s in range(0, lat.shape[0], chunk):
            pis = np.asarray(self.support.to_constrained(jnp.asarray(lat[s:s + chunk])))
            pis = pis.reshape(pis.shape[0], -1)
            if plans is None:
                plans = np.empty((lat.shape[0], pis.shape[-1]))
            plans[s:s + chunk] = pis
        return plans

    def initialize_diverse_chains(self, keys):
        denom = max(self.parallel_chains - 1, 1)
        if self.initial_plan is not None:
            # Seed each chain at a slightly perturbed version of the supplied plan (e.g. the
            # EOT/UOT Sinkhorn solution), mapped to the latent space -- starting inside the
            # mode's basin rather than the high-entropy interior. The per-chain perturbation
            # grows mildly across chains to keep enough spread for R-hat to be meaningful.
            y0 = self.support.to_unconstrained(self.initial_plan.reshape(1, -1)).reshape(self.shape,)
            
            lo, hi = self.warm_start_sigma
            def _one_chain(key, index):
                sigma = lo + (hi - lo) * index / denom
                return y0 + sigma * jax.random.normal(key, (self.shape,))
        else:
            # Diffuse fallback: increasing scale for wider exploration (single-chain safe).
            def _one_chain(key, index):
                scale = 0.1 + 0.99 * index / denom
                return jax.random.truncated_normal(key, lower=1e-6, upper=1, shape=(self.shape,)) * scale

        indices = jnp.arange(self.parallel_chains)
        return jax.vmap(_one_chain)(keys, indices)

    def _run_segment(self, num_steps, params, adapt_step, mass_diag):
        """One vmapped inference segment; returns (carry, stacked) with fresh per-chain keys."""
        self.init_key, sub_key = jax.random.split(self.init_key)
        keys = jax.device_put(jax.random.split(sub_key, self.parallel_chains), SHARDING)
        return jax.vmap(self.inference_loop, in_axes=(None, 0, 0, None, None))(
            num_steps, params, keys, adapt_step, mass_diag)

    @staticmethod
    def _window_mass(stacked_states, window_len, prior_strength=5.0):
        """Diagonal mass from ONE window: mean of within-chain variances, regularized.

        Within-chain (var over steps, then mean over chains), never pooled: by the law of
        total variance, pooling adds the between-chain dispersion of the means, which during
        warm-up measures initialization spread and unconverged transient (historically: gauge
        drift), not target variance. The estimate is shrunk toward its scalar mean with
        weight n/(n + prior_strength) so a short window degrades toward an isotropic-but-
        scaled mass instead of a noisy one (per-coordinate relative noise ~ sqrt(2/ESS_w)).
        """
        lat = stacked_states.stacked_latent_states           # (C, steps, 1, m)
        v = jnp.mean(jnp.var(lat, axis=1), axis=0)           # (1, m)
        w = window_len / (window_len + prior_strength)
        return w * v + (1.0 - w) * jnp.mean(v)

    def _warmup_windows(self):
        """Split warm_up_steps into (init_buffer, [expanding windows], term_buffer)."""
        W = self.warm_up_steps
        buf = max(1, W // 6)
        interior = max(1, W - 2 * buf)
        wins, size = [], max(1, interior // 8)
        while sum(wins) + size < interior:
            wins.append(size)
            size *= 2
        if wins:
            wins[-1] += interior - sum(wins)   # the LAST window absorbs the remainder: the
        else:                                  # final mass must come from the longest window
            wins = [interior]
        return buf, wins, buf

    def inference_loop_multiple_chains(self) -> Sequence[StackedMALAStates]:
        # * Initialize randomly the chains in parallel across the mesh. Ensure diversity of initial states.
        self.init_key, new_key = jax.random.split(self.init_key)
        new_keys = jax.random.split(new_key, self.parallel_chains)
        initial_states = self.initialize_diverse_chains(new_keys)
        jax.debug.print(f"Initial states shape: {initial_states.shape}")

        step_sizes = self.step_size * jnp.ones((self.parallel_chains, 1))
        params = jax.device_put((initial_states, step_sizes), SHARDING)
        ones_mass = jnp.ones((1, self.shape))

        if self.warmup == "legacy":
            # Historical flow, verbatim: warm-up with FROZEN step and identity mass; mass =
            # variance POOLED over chains and steps; Robbins-Monro then adapts the step
            # through burn-in AND retained draws. Kept only for E-series reproduction.
            carry, warm_stacked = self._run_segment(self.warm_up_steps, params,
                                                    False, ones_mass)
            mass_diag = jnp.var(warm_stacked.stacked_latent_states, axis=(0, 1))
            params = jax.device_put((carry.state, carry.step_size), SHARDING)
            _, mix_stacked_states = self._run_segment(self.tot_num_samples, params,
                                                      True, mass_diag)
            return mix_stacked_states

        # Windowed warm-up (Stan-shaped): the step adapts CONTINUOUSLY toward TARGET_ACCEPT
        # under the current mass through all of warm-up; the mass updates at window closes
        # from that window's within-chain variances; a terminal buffer re-settles the step
        # under the frozen final mass; the retained stage then runs a FIXED kernel (no
        # adaptation -- draws from a time-varying kernel are not MH-correct).
        init_buf, windows, term_buf = self._warmup_windows()
        mass_diag = ones_mass
        carry, _ = self._run_segment(init_buf, params, True, mass_diag)
        for win in windows:
            params = jax.device_put((carry.state, carry.step_size), SHARDING)
            carry, stacked = self._run_segment(win, params, True, mass_diag)
            mass_diag = self._window_mass(stacked, win)
        params = jax.device_put((carry.state, carry.step_size), SHARDING)
        carry, _ = self._run_segment(term_buf, params, True, mass_diag)

        steps = np.asarray(jax.device_get(carry.step_size)).reshape(-1)
        mass_host = np.asarray(jax.device_get(mass_diag)).reshape(-1)
        print(f"warmup[windowed] {init_buf}/{windows}/{term_buf}: "
              f"step med {np.median(steps):.2e} [{steps.min():.2e}, {steps.max():.2e}] | "
              f"mass cond {mass_host.max() / max(mass_host.min(), 1e-300):.1f}", flush=True)
        if np.any(steps <= 1.02e-8) or np.any(steps >= 0.98):
            print("warmup[windowed] WARNING: step clamp [1e-8, 1] is BINDING -- adaptation "
                  "failed; retained-phase acceptance will not match TARGET_ACCEPT.", flush=True)

        params = jax.device_put((carry.state, carry.step_size), SHARDING)
        _, mix_stacked_states = self._run_segment(self.tot_num_samples, params,
                                                  False, mass_diag)
        return mix_stacked_states

    def preconditioned_langevin_update(self, curr_pos, grad, step_size, mass_diag, key):
        """
        Performs a preconditioned Langevin update step, which is a combination of a gradient ascent step and a Gaussian noise term."""

        z = jax.random.normal(key, shape=(1, self.shape))
        step = step_size * mass_diag * grad
        noise = jnp.sqrt(2 * step_size * mass_diag) * z
        return curr_pos + step + noise

    @jax.jit(static_argnums=(0,1,4))
    def inference_loop(
        self,
        num_steps: int,
        init_params: Tuple[Array, Array],
        init_keys: Array,
        adapt_step: bool = False,
        mass_diag: Array = None,
    ) -> Tuple[MALAState, StackedMALAStates]:
        """
        Runs the inference loop of the Metropolis-Adjusted Langevin Algorithm (MALA) transition kernel.

        Args:
            num_steps (int): the number of steps to run the inference loop.
            init_params (Tuple[Array, Array]): the initial parameters for the inference loop, including the initial state and step size.
            init_keys (Array): the initial random keys for the inference loop.
            adapt_step (bool, optional): whether the Robbins-Monro step-size adaptation is
                active in this segment (the counter -- hence the learning rate (1/ct)^0.6 --
                restarts per segment, giving a hot re-adaptation after each mass update).
            mass_diag (Array): the diagonal of the mass matrix, always supplied by the
                caller (identity for the segments that precede the first estimate).

        """

        def langevin_step(current_state: MALAState, _):
            """Runs one step of Langevin Monte Carlo with a Metropolis-Hastings (MH) correction step.

            Args:
                current_x (MALAState): the current position.

            Returns:
                Array: the next position.
            """
            current_x = current_state.state
            current_ct = current_state.counter
            key = current_state.rng_keys
            step_size = current_state.step_size
            num_accepted_samples = current_state.num_accepted_samples
            avg_mh_ratio = current_state.avg_mh_ratio
            current_ct += 1
            key, sub_key = jax.random.split(key)

            # Compute current gradient
            current_grad = self.latent_score_function(current_x)
            # jax.debug.print("Current gradient: {x}", x=current_grad)

            # Propose a new sample (mass_diag is closed over; always caller-supplied)
            proposed_x, latent_log_prob_prop, latent_log_prob_curr, metrop_hastings_ratio = self.compute_metropolis_hastings_ratio(current_x, current_grad, step_size, mass_diag, sub_key)

            # MH Adjustment
            key, sub_key = jax.random.split(key)
            uniform_samp = jax.random.uniform(sub_key, (1,))[0]
            state = jnp.where(uniform_samp <= metrop_hastings_ratio, proposed_x, current_x)
            latent_log_prob_state = jnp.where(uniform_samp <= metrop_hastings_ratio, latent_log_prob_prop, latent_log_prob_curr)

            num_accepted_samples += jnp.where(uniform_samp <= metrop_hastings_ratio, 1, 0)

            # Update the exponential moving avg. of the MH acceptance ratio. The smoothing parameter alpha controls the sensitivity of the step size adaptation
            # to recent MH ratios.
            # When alpha = 1, only the most recent MH ratio is considered, while when alpha = 0, the average MH ratio is not updated at all and remains constant
            # across iterations.
            avg_mh_ratio = jnp.where(current_ct == 1, metrop_hastings_ratio, self.alpha * metrop_hastings_ratio + (1 - self.alpha) * avg_mh_ratio)
            avg_mh_ratio = jnp.float32(avg_mh_ratio)
            # jax.debug.print("Current average MH ratio: {x}", x=avg_mh_ratio)

            # Robbins-Monro step-size adaptation, active only in segments where the caller
            # enabled it (warm-up under the windowed scheme; burn-in + retained under legacy).
            log_step_size = jnp.log(step_size)
            new_log_step_size = log_step_size + (1 / current_ct)**(0.6) * (avg_mh_ratio - TARGET_ACCEPT)
            log_step_size = jnp.where(adapt_step, new_log_step_size, log_step_size)
            step_size = jnp.exp(log_step_size)

            # * Clip the step size to avoid exploding values or very small values that would lead to numerical issues.
            # * The upper bound of 1 is chosen heuristically, while the lower bound of 1e-8 is chosen to avoid numerical
            #  issues in the log-probability and proposal distribution computations.
            # Positional bounds: numpy 2.0 renamed clip's `a_min`/`a_max` to `min`/`max` and jax
            # followed (the old names were removed in jax 0.10). Positional works on both.
            step_size = jnp.clip(step_size, 1e-8, 1)
            step_size = jnp.float32(step_size)
            # jax.debug.print("Current step size: {x}", x=step_size)

            carry_state = MALAState(state=state,
                                    counter=current_ct,
                                    rng_keys=key,
                                    step_size=step_size,
                                    num_accepted_samples=num_accepted_samples,
                                    avg_mh_ratio=avg_mh_ratio)

            return carry_state, (state, latent_log_prob_state, num_accepted_samples, avg_mh_ratio)
    
        initial_ct = 0
        avg_mh_ratio = jnp.expand_dims(jnp.float32(0.), axis=0)
        num_accepted_samples = jnp.expand_dims(jnp.int32(0), axis=0)
        initial_state = init_params[0].reshape(1, self.shape)
        initial_step_size = jnp.float32(init_params[1])
    
        # First stage warm-up without step size adaptation for diag. mass matrix estimation.
        # During this stage, the step size is kept fixed.
        init_mala_state = MALAState(state=initial_state,
                                    counter=initial_ct,
                                    rng_keys=init_keys,
                                    step_size=initial_step_size,
                                    num_accepted_samples=num_accepted_samples,
                                    avg_mh_ratio=avg_mh_ratio,
                                    latent_log_prob_state=None)

        carry_state, (stacked_states, stacked_latent_log_prob_states, stacked_num_accepted_samples, stacked_avg_mh_ratio) = jax.lax.scan(langevin_step, init_mala_state, None, num_steps)

        summary_stacked_states = StackedMALAStates(stacked_latent_states=stacked_states,
                                                  stacked_latent_log_prob_states=stacked_latent_log_prob_states,
                                                  stacked_num_accepted_samples=stacked_num_accepted_samples,
                                                  stacked_avg_mh_ratios=stacked_avg_mh_ratio)

        # FIXME: returning both the stacked and the last carry states is redundant. Use only the stacked states and extract the last state from it if needed.
        return carry_state, summary_stacked_states
    
    def compute_metropolis_hastings_ratio(self, current_x: Array, current_grad: Array, step_size: float, mass_diag: Array, sub_key: Array) -> Tuple:

        proposed_x = self.preconditioned_langevin_update(current_x, current_grad, step_size, mass_diag, sub_key)
        log_prop_dist_forward =  self.log_proposal_dist(x=current_x,
                                        y=proposed_x,
                                        curr_step_size=step_size,
                                        mass_diag=mass_diag)
        log_prop_dist_backward = self.log_proposal_dist(x=proposed_x,
                                        y=current_x,
                                        curr_step_size=step_size,
                                        mass_diag=mass_diag)
        
        # MH log-ratio
        latent_log_prob_prop = self.latent_log_prob_function(proposed_x)
        latent_log_prob_curr = self.latent_log_prob_function(current_x)
        mh_log_ratio = (
                latent_log_prob_prop
                + log_prop_dist_forward
                - latent_log_prob_curr
                - log_prop_dist_backward
            )
        mh_log_ratio = jnp.where(jnp.isfinite(mh_log_ratio), mh_log_ratio, -jnp.inf)
        metrop_hastings_ratio = jnp.exp(mh_log_ratio)
        metrop_hastings_ratio = jnp.where(metrop_hastings_ratio <= 1, metrop_hastings_ratio, 1)
        metrop_hastings_ratio = jnp.squeeze(metrop_hastings_ratio)

        return proposed_x, latent_log_prob_prop, latent_log_prob_curr, metrop_hastings_ratio

    def log_proposal_dist(self, x: Array, y: Array, curr_step_size: float, mass_diag: Array) -> float:
        r"""Computes the log transition probability density function at x given y:
        log(q(x|y)) \propto - \frac{1}{2 * step**2} \|x - y - 0.5 * step_size * gradient(log_probability_fn)(y)\|**2

        Args:
            new_x (np.ndarray): the candidate position.
            x (np.ndarray): the current position.
            step (float): the step size in the Langevin diffusion model.

        Returns:
            _type_: _description_
        """
        logger.info("Computing the log transition kernel")
        diff = x - y - curr_step_size * mass_diag * self.latent_score_function(y)
        return - 0.5 * jnp.sum(jnp.square(diff) / (2 * curr_step_size * mass_diag))


class HFPDOTHyperprior:
    def __init__(
        self, mu_0, nu_0, lambda_1, lambda_2, lambda_I_1, lambda_I_2, cost_fn, epsilon,
        support: Literal["simplex", "positive_orthant"] = "simplex",
        lambda_pi: float = 1.0
    ):
        self.mu_0 = jnp.asarray(mu_0)
        self.nu_0 = jnp.asarray(nu_0)
        self.cost_fn = cost_fn
        self.epsilon = epsilon
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_I_1 = lambda_I_1
        self.lambda_I_2 = lambda_I_2
        # Weight of the ideal-design term gKL(pi || pi_I). 1.0 is the model; on the
        # positive orthant pi_I is the unnormalised Gibbs kernel, so the term also
        # anchors total mass -- values below 1 are attribution diagnostics for that pull.
        self.lambda_pi = lambda_pi
        self.support = support
        self.II = len(self.mu_0)
        self.JJ = len(self.nu_0)
        self.pi_I = jnp.exp(- self.cost_fn / self.epsilon)
        self.pi_I = jnp.reshape(self.pi_I, (1, -1))
        self.init_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

    def sinkhorn_init(self, reg_m: float | None = None, num_iter: int = 1000) -> Array:
        """Sinkhorn solution used to seed the sampler near the mode, returned as (1, II*JJ).

        Balanced ("simplex"): the entropic-OT coupling (``ot.sinkhorn``) matching mu_0, nu_0.
        Unbalanced ("positive_orthant"): the unbalanced-OT plan (``ot.sinkhorn_unbalanced``)
        with KL-relaxed marginals; ``reg_m`` (relaxation strength) defaults to the
        hyperprior's marginal lambdas.

        Uses POT (mature, log-domain stabilized for small epsilon). POT's jax backend is not
        available with current jax, so this one-time preprocessing runs on CPU (numpy) -- a
        negligible cost; the sampling loop itself stays on the accelerator. Import is lazy so
        the generic sampler core does not depend on POT.
        """
        import numpy as np
        import ot

        a = np.asarray(self.mu_0, dtype=float)
        b = np.asarray(self.nu_0, dtype=float)
        M = np.asarray(self.cost_fn, dtype=float).reshape(self.II, self.JJ)
        if self.support == "simplex":
            plan = ot.sinkhorn(a, b, M, reg=self.epsilon, numItermax=num_iter)
        else:  # positive_orthant
            if reg_m is None:
                reg_m = [self.lambda_1 + self.lambda_I_1, self.lambda_2 + self.lambda_I_2]
            plan = ot.sinkhorn_unbalanced(a, b, M, reg=self.epsilon, reg_m=reg_m, numItermax=num_iter)
        return jnp.asarray(plan).reshape(1, -1)

    def balanced_hyperprior_log_prob_fun(self, pi: Array) -> float:
        """Evaluates the log-hyperprior at the state $\pi$, a vector of size (1, II x JJ).
        Note that the input is log_pi, the log of pi, to ensure the positivity constraint on pi. The log-hyperprior is derived via a change of variable
        and extends the originally -- positivity-constrained -- distribution to $\mathbb{R}^{II x JJ}$ with the appropriate change of variable.

        Args:
            pi (Iterable): _description_

        Returns:
            float: _description_
        """
        pi_mat = jnp.reshape(pi, (self.II, self.JJ))
        mu = jnp.sum(pi_mat, axis=1)
        nu = jnp.sum(pi_mat, axis=0)

        kl1 = - (self.lambda_1 + self.lambda_I_1) * HFPDOTHyperprior.shifted_kl_div(
            mu, self.mu_0
        )
        kl2 = - (self.lambda_2 + self.lambda_I_2) * HFPDOTHyperprior.shifted_kl_div(
            nu, self.nu_0
        )
        kl3 = - self.lambda_pi * HFPDOTHyperprior.shifted_kl_div(pi, self.pi_I)

        return kl1 + kl2 + kl3
    
    def unbalanced_hyperprior_log_prob_fun(self, pi: Array) -> float:
        """Evaluates the log-hyperprior at the state $\pi$, a vector of size (1, II x JJ).
        Note that the input is log_pi, the log of pi, to ensure the positivity constraint on pi. The log-hyperprior is derived via a change of variable
        and extends the originally -- positivity-constrained -- distribution to $\mathbb{R}^{II x JJ}$ with the appropriate change of variable.

        Args:
            pi (Iterable): _description_

        Returns:
            float: _description_
        """
        pi_mat = jnp.reshape(pi, (self.II, self.JJ))
        mu = jnp.sum(pi_mat, axis=1)
        nu = jnp.sum(pi_mat, axis=0)

        kl1 = - (self.lambda_1 + self.lambda_I_1) * HFPDOTHyperprior.generalized_kl_div(
            mu, self.mu_0
        )
        kl2 = - (self.lambda_2 + self.lambda_I_2) * HFPDOTHyperprior.generalized_kl_div(
            nu, self.nu_0
        )
        kl3 = - self.lambda_pi * HFPDOTHyperprior.generalized_kl_div(pi, self.pi_I)

        return kl1 + kl2 + kl3
    
    def hyperprior_log_prob_fun(self, pi: Array) -> float:
        """Evaluates the log-hyperprior at the state $\pi$, a vector of size (1, II x JJ).
        Note that the input is log_pi, the log of pi, to ensure the positivity constraint on pi. The log-hyperprior is derived via a change of variable
        and extends the originally -- positivity-constrained -- distribution to $\mathbb{R}^{II x JJ}$ with the appropriate change of variable.

        Args:
            pi (Iterable): _description_

        Returns:
            float: _description_
        """
        if self.support == "simplex":
            return self.balanced_hyperprior_log_prob_fun(pi)
        elif self.support == "positive_orthant":
            return self.unbalanced_hyperprior_log_prob_fun(pi)
        else:
            raise ValueError(f"Unsupported support type: {self.support}. Must be 'simplex' or 'positive_orthant'.")

    @staticmethod
    def safe_log(p: Array, epsilon=1e-15)->Array:
        return jnp.log(p + epsilon)
    
    @staticmethod
    def generalized_kl_div(p: Array, q: Array, epsilon: float = 1e-9) -> Array:
        """Computes the generalized KL divergence between two positive measures p and q.
        Accounts for the case where p and q are not normalized to sum to 1, by balancing the mass.
        Args:
            p (NDArray): a probability vector of size (1, II)
            q (NDArray): a probability vector of size (1, JJ)
            epsilon (float): the smoothing parameter in the KL.

        Returns:
            float: _description_
        """
        log_ratio = jnp.log((p + epsilon) / (q + epsilon))
        return jnp.dot((p + epsilon), log_ratio.T) - jnp.sum(p) + jnp.sum(q)
        

    @staticmethod
    def shifted_kl_div(p: Array, q: Array, epsilon: float = 1e-9) -> Array:
        """Implements the shifted version of the KL divergence, as described in https://arxiv.org/pdf/2312.13021v2,
        with a static choice of the smoothing parameter, $\epsilon$.
        Args:
            p (NDArray): a probability vector of size (1, II)
            q (NDArray): a probability vector of size (1, JJ)
            epsilon (float): the shift parameter in the KL.

        Returns:
            float: _description_
        """
        log_ratio = jnp.log((p + epsilon) / (q + epsilon))
        return jnp.dot((p + epsilon), log_ratio.T)

    @staticmethod
    def _softmax(state: Array):
        return jax.nn.softmax(state, axis=-1)

    def hyperprior_score_fun(self, pi: Array, epsilon: float = 1e-9) -> Array:
        """Score of the HFPD-OT hyperprior at pi (flattened, shape (1, II*JJ)).

        Each term is the generalized-KL gradient log((p+eps)/(q+eps)) with NO
        additive +1. One score serves BOTH regimes:

          - Unbalanced (positive orthant): the generalized-KL log-prob has gradient
            log((p+eps)/(q+eps)); this is its exact score.
          - Balanced (simplex): the (shifted-)KL log-prob gradient would carry a +1
            per term, i.e. this score plus a *global constant* c. But the simplex
            lift only uses the score through the centering F*(s - <F,s>); since
            sum(F)=1 that projection annihilates any constant added to s, so the
            missing +1 is invisible and this score is still exact on the simplex.

        The positive orthant has no such projection (latent_score = pi*s + 1), so
        the +1 would survive as a spurious pi*c term -- hence it is dropped here.

        Args:
            pi (NDArray): flattened probability vector of shape (1, II*JJ).
        """
        assert pi.shape[1] == self.II * self.JJ
        pi_mat = pi.reshape((self.II, self.JJ))
        pi_I_mat = self.pi_I.reshape(self.II, self.JJ)
        mu = jnp.sum(pi_mat, axis=1)
        nu = jnp.sum(pi_mat, axis=0)
        logger.info(f"Shape of the first marginal: {mu.shape}")
        logger.info(f"Shape of the second marginal: {nu.shape}")
        assert mu.shape == self.mu_0.shape
        assert nu.shape == self.nu_0.shape
        logger.info("Initiating the gradient computation")

        grad_mu = -(self.lambda_1 + self.lambda_I_1) * jnp.log((mu + epsilon) / (self.mu_0 + epsilon))
        grad_nu = -(self.lambda_2 + self.lambda_I_2) * jnp.log((nu + epsilon) / (self.nu_0 + epsilon))
        grad_pi = -self.lambda_pi * jnp.log((pi_mat + epsilon) / (pi_I_mat + epsilon))

        grad = grad_pi + jnp.expand_dims(grad_mu, axis=1) + jnp.expand_dims(grad_nu, axis=0)
        return grad.reshape(pi.shape)