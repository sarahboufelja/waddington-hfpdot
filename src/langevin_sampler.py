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
from supports import make_support, LogProbFn, ScoreFn
from mcmc_diagnostics import MCMCDiagnostics

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
        support: Literal["simplex", "positive_orthant", "unconstrained"] = "unconstrained",
        num_samples: int = 8_000,
        num_burnin: int = 8_000,
        warm_up_steps: int = 500,
        step_size: float = 0.1,
        num_parallel_chains: int = jax.local_device_count(),
        seed: int = 0,
        initial_plan: Array | None = None,
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
        self.init_key = jax.random.key(self.seed)
        self.alpha = alpha ## smoothing factor of the exponential moving average
        self.num_samples = num_samples
        self.tot_num_samples = self.num_samples + self.num_burnin

        self.support = make_support(
            support,
            radial_log_prob_fn=radial_log_prob_fn,
            radial_score_fn=radial_score_fn,
        )

        self.latent_log_prob_function = lambda x: self.support.latent_log_prob(self.target_log_prob_function, x)
        self.latent_score_function = lambda x: self.support.latent_score(self.target_score_function, x)

        # Convergence diagnostics operate on the latent sampling space, per chain.
        self._diagnostics = MCMCDiagnostics(log_prob_fn=self.latent_log_prob_function)


    def sample(
        self,
        with_diagnostics: bool = True,):
        
        # Sample with multiple chains and run diagnostics
        samples, num_accepted_samples, mh_log_ratio = self.inference_loop_multiple_chains()

        # Remove the burn-in samples: (num_chains, num_kept, 1, sample_dim).
        samples = samples[:, self.num_burnin:]

       # Drop the singleton dimension: (num_chains, num_kept, sample_dim).
        latent_samples = jnp.squeeze(samples, axis=2)
        
        # Compute diagnostic metrics in the latent spaces
        diag = self._diagnostics.summarize(latent_samples) if with_diagnostics else None

        # First flatten the samples then project back to the constrained target space (simplex / positive orthant / identity).
        flattened_samples = jnp.reshape(latent_samples, (-1, self.shape))
        constrained_samples = self.support.to_constrained(flattened_samples)
        
        return constrained_samples, num_accepted_samples, diag, mh_log_ratio

    def initialize_diverse_chains(self, keys):
        denom = max(self.parallel_chains - 1, 1)
        if self.initial_plan is not None:
            # Seed each chain at a slightly perturbed version of the supplied plan (e.g. the
            # EOT/UOT Sinkhorn solution), mapped to the latent space -- starting inside the
            # mode's basin rather than the high-entropy interior. The per-chain perturbation
            # grows mildly across chains to keep enough spread for R-hat to be meaningful.
            y0 = self.support.to_unconstrained(self.initial_plan.reshape(1, self.shape)).reshape(self.shape)

            def _one_chain(key, index):
                sigma = 0.1 + 0.2 * index / denom
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
        # * carry layout: (state, counter, key, step_size, num_accepted, avg_mh_ratio); we reuse state[0] and step_size[3].
        main_init_params = (warm_carry_state[0], warm_carry_state[3])
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

    def preconditioned_langevin_update(self, curr_pos, grad, step_size, mass_diag, key):
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
    ) -> tuple:
        """
        Runs the inference loop of the Metropolis-Adjusted Langevin Algorithm (MALA) transition kernel.

        Args:
            num_steps (int): the number of steps to run the inference loop.
            init_params (Array): the initial parameters for the inference loop, including the initial state and step size.
            init_keys (Array): the initial random keys for the inference loop.
            warm_up (bool, optional): whether to perform the warm-up phase. Defaults to True.
            mass_diag (Array, optional): the diagonal of the mass matrix. Defaults to None.
        
        """

        def langevin_step(current_state: tuple, _):
            """Runs one step of Langevin Monte Carlo with a Metropolis-Hastings (MH) correction step.

            Args:
                current_x (Array): the current position.

            Returns:
                Array: the next position.
            """
            nonlocal mass_diag
            current_x = current_state[0]
            current_ct = current_state[1]
            key = current_state[2]
            step_size = current_state[3]
            num_accepted_samples = current_state[4]
            avg_mh_ratio = current_state[5]
            current_ct += 1
            key, sub_key = jax.random.split(key)

            # Compute current gradient
            current_grad = self.latent_score_function(current_x)
            # jax.debug.print("Current gradient: {x}", x=current_grad)

            # Propose a new sample
            mass_diag = mass_diag if not warm_up else jnp.ones_like(current_x)
            proposed_x = self.preconditioned_langevin_update(current_x, current_grad, step_size, mass_diag, sub_key)

            # MH log-ratio
            mh_log_ratio = (
                self.latent_log_prob_function(proposed_x)
                + self.log_proposal_dist(x=current_x,
                                         y=proposed_x,
                                         curr_step_size=step_size,
                                         mass_diag=mass_diag)
                - self.latent_log_prob_function(current_x)
                - self.log_proposal_dist(x=proposed_x,
                                         y=current_x,
                                         curr_step_size=step_size,
                                         mass_diag=mass_diag)
            )
            mh_log_ratio = jnp.where(jnp.isfinite(mh_log_ratio), mh_log_ratio, -jnp.inf)
            metrop_hastings_ratio = jnp.exp(mh_log_ratio)
            metrop_hastings_ratio = jnp.where(metrop_hastings_ratio <= 1, metrop_hastings_ratio, 1)
            metrop_hastings_ratio = jnp.squeeze(metrop_hastings_ratio)

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

            return (state, current_ct, key, step_size, num_accepted_samples, avg_mh_ratio), (state, num_accepted_samples, avg_mh_ratio)

        initial_ct = 0
        avg_mh_ratio = jnp.expand_dims(jnp.float32(0.), axis=0)
        num_accepted_samples = jnp.expand_dims(jnp.int32(0), axis=0)
        initial_state = init_params[0].reshape(1, self.shape)
        initial_step_size = jnp.float32(init_params[1])
    
        # First stage warm-up without step size adaptation for diag. mass matrix estimation.
        # During this stage, the step size is kept fixed.
        carry_state, (states, num_accepted_samples, avg_mh_ratios) = jax.lax.scan(langevin_step, (initial_state,
                                                                                                  initial_ct,
                                                                                                  init_keys,
                                                                                                  initial_step_size,
                                                                                                  num_accepted_samples,
                                                                                                  avg_mh_ratio),
                                                                                                    None,
                                                                                                    num_steps)

        return states, num_accepted_samples, avg_mh_ratios, carry_state

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