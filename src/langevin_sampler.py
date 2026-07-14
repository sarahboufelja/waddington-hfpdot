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
from typing import Tuple, Literal
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

@dataclass
class FinalMalaState:
    """Holds the final state of the MALA sampler after running inference_loop."""
    samples: Array
    num_accepted_samples: Array
    diagnostics: DiagnosticsSummary
    mh_log_ratios: Array

# Registered as a pytree so it can be the carry of jax.lax.scan / vmap (all fields are
# dynamic arrays -> data_fields; no static metadata).
@partial(jax.tree_util.register_dataclass,
         data_fields=["state", "counter", "rng_keys", "step_size",
                      "num_accepted_samples", "avg_mh_ratio"],
         meta_fields=[])
@dataclass
class MALAState:
    """Holds the intermediate state of the MALA sampler while running inference_loop."""
    state: Array
    counter: Array
    rng_keys: Array
    step_size: Array
    num_accepted_samples: Array
    avg_mh_ratio: Array

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
        with_diagnostics: bool = True,):
        
        # Sample with multiple chains and run diagnostics
        samples, num_accepted_samples, mh_log_ratio = self.inference_loop_multiple_chains()

        # Remove the burn-in samples: (num_chains, num_kept, 1, sample_dim).
        samples = samples[:, self.num_burnin:]

       # Drop the singleton dimension: (num_chains, num_kept, sample_dim).
        latent_samples = jnp.squeeze(samples, axis=2)  # Shape = (C, D, m) where m = sample_dim
        
        constrained_samples = self.support.to_constrained(latent_samples)  # Shape = (C, D, m) where m = sample_dim

        # Compute diagnostic metrics in the transformed spaces (pi-space)
        # Convergence diagnostics operate on the latent sampling space, per chain.
        # eBFMI needs the log-prob, so we pass it to the diagnostics object; R-hat/ESS are gauge-invariant on pi, so we don't need a log-prob for that.
        self._diagnostics = MCMCDiagnostics(constrained_chains=constrained_samples,
                                            latent_chains=latent_samples,
                                            log_prob_fn=self.latent_log_prob_function)
        diag = self._diagnostics.summarize() if with_diagnostics else None

        return FinalMalaState(
            samples=constrained_samples,
            num_accepted_samples=num_accepted_samples,
            diagnostics=diag,
            mh_log_ratios=mh_log_ratio
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

    def inference_loop_multiple_chains(self):
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
        (warm_pmap_states,
        warm_pmap_num_accepted_samps,
        warm_pmap_mh_ratio,
        warm_carry_state) = jax.vmap(self.inference_loop, in_axes=(None, 0, 0, None))(self.warm_up_steps,
                                                                                      sharded_params,
                                                                                      sharded_keys,
                                                                                      True)
        # * Compute the variance of each variable across all chains and all warm-up samples to estimate the diagonal
        # * of the mass matrix.
        mass_diag = jnp.var(warm_pmap_states, axis=(0, 1))

        # * Second stage is the main sampling stage with Robbins-Monro step size adaptation, using the estimated mass matrix.
        # * Continue from the warm-up's final state and step size to ensure continuity between the two stages.
        # * Reuse the warm-up's final state and step size (MALAState is a registered pytree).
        main_init_params = (warm_carry_state.state, warm_carry_state.step_size)
        sharded_params = jax.device_put(main_init_params, SHARDING)
        self.init_key, sub_key = jax.random.split(self.init_key)
        sharded_keys = jax.random.split(sub_key, self.parallel_chains)
        sharded_keys = jax.device_put(sharded_keys, SHARDING)
        (pmap_states,
         pmap_num_accepted_samps,
         pmap_mh_ratio,
         _) = jax.vmap(self.inference_loop, in_axes=(None, 0, 0, None, None))(self.tot_num_samples,
                                                                             sharded_params,
                                                                             sharded_keys,
                                                                             False,
                                                                             mass_diag)
        
        return pmap_states, pmap_num_accepted_samps, pmap_mh_ratio

    def preconditioned_langevin_update(self, curr_pos, grad, step_size, mass_diag, key, support_name):
        """
        Performs a preconditioned Langevin update step, which is a combination of a gradient ascent step and a Gaussian noise term."""

        if self.support.name == "low_rank":
            # For low-rank support, we use the support's custom Langevin update.
            return self.support.propose_gauge_constrained_step(curr_pos, step_size)

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
    ) -> Tuple:
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

            proposed_x, metrop_hastings_ratio = self.compute_metropolis_hastings_ratio(current_x, current_grad, step_size, mass_diag, sub_key)

            # MH Adjustment
            key, sub_key = jax.random.split(key)
            uniform_samp = jax.random.uniform(sub_key, (1,))[0]
            state = jnp.where(uniform_samp <= metrop_hastings_ratio, proposed_x, current_x)

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

            return carry_state, (state, num_accepted_samples, avg_mh_ratio)
    
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
                                    avg_mh_ratio=avg_mh_ratio)
        carry_state, (stacked_states, stacked_num_accepted_samples, stacked_avg_mh_ratios) = jax.lax.scan(langevin_step, init_mala_state, None, num_steps)

        return stacked_states, stacked_num_accepted_samples, stacked_avg_mh_ratios, carry_state
    
    def compute_metropolis_hastings_ratio(self, current_x: Array, current_grad: Array, step_size: float, mass_diag: Array, sub_key: Array) -> Tuple:

        if self.support.name != "low_rank":
            proposed_x = self.preconditioned_langevin_update(current_x, current_grad, step_size, mass_diag, sub_key, self.support.name)
            log_prop_dist_forward =  self.log_proposal_dist(x=current_x,
                                         y=proposed_x,
                                         curr_step_size=step_size,
                                         mass_diag=mass_diag)
            log_prop_dist_backward = self.log_proposal_dist(x=proposed_x,
                                            y=current_x,
                                            curr_step_size=step_size,
                                            mass_diag=mass_diag)
        
        else:
            proposed_x_summary = self.support.propose_bal_gauge_constrained_step(current_x, step_size)
            log_prop_dist_forward = self.support.evaluate_bal_gauge_log_proposal_density(proposed_x_summary)
            backward_noise, backward_velocity = self.support.invert_retraction(proposed_x_summary.state, current_x)
            log_prop_dist_backward = self.support.evaluate_bal_gauge_log_proposal_density(self.support.StateSummary(state=current_x,
                                                                                                                    matching_noise=backward_noise,
                                                                                                                    velocities=backward_velocity))

        # MH log-ratio
        mh_log_ratio = (
                self.latent_log_prob_function(proposed_x)
                + log_prop_dist_forward
                - self.latent_log_prob_function(current_x)
                - log_prop_dist_backward
            )
        mh_log_ratio = jnp.where(jnp.isfinite(mh_log_ratio), mh_log_ratio, -jnp.inf)
        metrop_hastings_ratio = jnp.exp(mh_log_ratio)
        metrop_hastings_ratio = jnp.where(metrop_hastings_ratio <= 1, metrop_hastings_ratio, 1)
        metrop_hastings_ratio = jnp.squeeze(metrop_hastings_ratio)

        return proposed_x, metrop_hastings_ratio

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
        if self.support.name == "low_rank":
            return self.support.evaluate_bal_gauge_log_proposal_density(x, curr_step_size)

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