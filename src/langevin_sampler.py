from dataclasses import dataclass
from functools import partial
import logging
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
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

        # Compute diagnostic metrics in the transformed spaces (pi-space)
        # Convergence diagnostics operate on the latent sampling space, per chain.
        # eBFMI needs the log-prob, so we pass it to the diagnostics object; R-hat/ESS are gauge-invariant on pi, so we don't need a log-prob for that.
        self._diagnostics = MCMCDiagnostics(constrained_chains=constrained_samples,
                                            latent_log_prob_states=latent_log_prob_states, # eBFMI needs the log-prob, so we pass it to the diagnostics object; R-hat/ESS are gauge-invariant on pi, so we don't need a log-prob for that.
                                            )
        diag = self._diagnostics.summarize() if with_diagnostics else None

        return MalaChainSummary(
            samples=constrained_samples,
            diagnostics=diag,
            pi_coord_idx=pi_coord_idx,
            stacked_mala_states=mix,   # full archive (incl. burn-in): latent theta + acceptance counts
        )

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

    def inference_loop_multiple_chains(self) -> Sequence[StackedMALAStates]:
        self.init_key, new_key = jax.random.split(self.init_key)
        sample_keys = jax.random.split(new_key, self.parallel_chains)
        # * Sharding the keys across devices to run the chains in parallel.
        sharded_keys = jax.device_put(sample_keys, SHARDING)

        # * Initialize randomly the chains in parallel across the mesh. Ensure diversity of initial states.
        self.init_key, new_key = jax.random.split(self.init_key)
        new_keys = jax.random.split(new_key, self.parallel_chains)
        initial_states = self.initialize_diverse_chains(new_keys)
        jax.debug.print(f"Initial states shape: {initial_states.shape}")

        step_sizes = self.step_size * jnp.ones((self.parallel_chains, 1))
        params_to_shard = (initial_states, step_sizes)
        sharded_params = jax.device_put(params_to_shard, SHARDING)

        # First stage is a warm-up stage without step adaptation.
        (warm_carry_states,
         warm_stacked_states) = jax.vmap(self.inference_loop, in_axes=(None, 0, 0, None))(self.warm_up_steps,
                                                                                          sharded_params,
                                                                                          sharded_keys,
                                                                                          True)
        # * Compute the variance of each variable across all chains and all warm-up samples to estimate the diagonal
        # * of the mass matrix. warm_stacked_states is a SINGLE vmapped StackedMALAStates whose fields
        # * carry a leading chain axis -> (C, warm_steps, 1, m); average over chains and steps.
        # TODO: confirm if the variance should be computed in the latent space versus the constrained space.
        # The current implementation computes the variance in the latent space.
        mass_diag = jnp.var(warm_stacked_states.stacked_latent_states, axis=(0, 1))

        # * Second stage is the main sampling stage with Robbins-Monro step size adaptation, using the estimated mass matrix.
        # * Continue from the warm-up's final state and step size to ensure continuity between the two stages.
        # * warm_carry_states is a SINGLE vmapped MALAState (fields batched over chains).
        init_states = warm_carry_states.state             # (C, 1, m)
        init_step_sizes = warm_carry_states.step_size     # (C, 1)
        main_init_params = (init_states, init_step_sizes)
        sharded_params = jax.device_put(main_init_params, SHARDING)
        self.init_key, sub_key = jax.random.split(self.init_key)
        sharded_keys = jax.random.split(sub_key, self.parallel_chains)
        sharded_keys = jax.device_put(sharded_keys, SHARDING)
        (_,
         mix_stacked_states) = jax.vmap(self.inference_loop, in_axes=(None, 0, 0, None, None))(self.tot_num_samples,
                                                                             sharded_params,
                                                                             sharded_keys,
                                                                             False,
                                                                             mass_diag)
        
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
        warm_up: bool = True,
        mass_diag: Array = None,
    ) -> Tuple[MALAState, StackedMALAStates]:
        """
        Runs the inference loop of the Metropolis-Adjusted Langevin Algorithm (MALA) transition kernel.

        Args:
            num_steps (int): the number of steps to run the inference loop.
            init_params (Tuple[Array, Array]): the initial parameters for the inference loop, including the initial state and step size.
            init_keys (Array): the initial random keys for the inference loop.
            warm_up (bool, optional): whether to perform the warm-up phase. Defaults to True.
            mass_diag (Array, optional): the diagonal of the mass matrix. Defaults to None.
        
        """

        def langevin_step(current_state: MALAState, _):
            """Runs one step of Langevin Monte Carlo with a Metropolis-Hastings (MH) correction step.

            Args:
                current_x (MALAState): the current position.

            Returns:
                Array: the next position.
            """
            nonlocal mass_diag
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

            # Propose a new sample
            mass_diag = mass_diag if not warm_up else jnp.ones_like(current_x)

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

            # Adapt the step size using a Robbins-Monroe process, after the warm-up phase is completed.
            log_step_size = jnp.log(step_size)
            new_log_step_size = log_step_size + (1 / current_ct)**(0.6) * (avg_mh_ratio - TARGET_ACCEPT)
            log_step_size = jnp.where(not warm_up, new_log_step_size, log_step_size)
            step_size = jnp.exp(log_step_size)

            # * Clip the step size to avoid exploding values or very small values that would lead to numerical issues.
            # * The upper bound of 1 is chosen heuristically, while the lower bound of 1e-8 is chosen to avoid numerical
            #  issues in the log-probability and proposal distribution computations.
            step_size = jnp.clip(step_size, a_min=1e-8, a_max=1)
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
        support: Literal["simplex", "positive_orthant"] = "simplex"
    ):
        self.mu_0 = jnp.asarray(mu_0)
        self.nu_0 = jnp.asarray(nu_0)
        self.cost_fn = cost_fn
        self.epsilon = epsilon
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_I_1 = lambda_I_1
        self.lambda_I_2 = lambda_I_2
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
        kl3 = - HFPDOTHyperprior.shifted_kl_div(pi, self.pi_I)

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
        kl3 = - HFPDOTHyperprior.generalized_kl_div(pi, self.pi_I)

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
        grad_pi = -jnp.log((pi_mat + epsilon) / (pi_I_mat + epsilon))

        grad = grad_pi + jnp.expand_dims(grad_mu, axis=1) + jnp.expand_dims(grad_nu, axis=0)
        return grad.reshape(pi.shape)