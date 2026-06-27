import numpy as np
import logging
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from datetime import date
from jax import Array
import jax.numpy as jnp
from jax.scipy.special import ndtri
import jax.scipy.stats as stats
from jax.scipy.special import logsumexp
from jax.experimental import mesh_utils
from typing import Tuple, Literal
from supports import make_support, LogProbFn, ScoreFn

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
        self.init_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
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
    
    
    def sample(
        self,
        with_diagnostics: bool = True,):
        
        # Sample with multiple chains and run diagnostics
        samples, num_accepted_samples, mh_log_ratio = self.inference_loop_multiple_chains()

        # Remove the burn-in samples: (num_chains, num_kept, 1, sample_dim).
        samples = samples[:, self.num_burnin:]

        # Flatten chains and samples into (num_chains * num_kept, sample_dim). These are
        # still latent (unconstrained) states; the projection to the target space is below.
        num_chains, num_kept = samples.shape[0], samples.shape[1]
        latent_samples = jnp.reshape(samples, (num_chains * num_kept, self.shape))

        # Diagnostics are assessed in the latent sampling space.
        diag = self.compute_mcmc_diagnostics(latent_samples) if with_diagnostics else None

        # Project back to the constrained target space (simplex / positive orthant / identity).
        constrained_samples = self.support.to_constrained(latent_samples)
        return constrained_samples, num_accepted_samples, diag, mh_log_ratio

    def initialize_diverse_chains(self, keys):
        # * Initialize the chains with an increasing scale for wider exploration.
        # * TODO: Use the entropy regularized solution of the unbalanced optimal transport problem to initialize the chains
        # in a more principled way and facilitate convergence to the target distribution.
        def _one_chain(key, index):
            scale = 0.1 + 0.99 * index / (self.parallel_chains - 1)
            return jax.random.truncated_normal(key,
                                               lower = 1e-6,
                                               upper = 1,
                                               shape=(self.shape,),) * scale
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
        # * Update the initial parameters with the final state of the warm-up stage to ensure continuity between the two stages.
        sharded_params = (warm_carry_state[0], warm_carry_state[3])
        sharded_params = jax.device_put(params_to_shard, SHARDING)
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

    
    def compute_ebfmi(self, samples: Array) -> float:
        """
        Computes the Bayesian Fraction of Missing Info for each chain, as described in https://arxiv.org/pdf/2202.05483.pdf.
        """
        energy_history = - jax.vmap(self.latent_log_prob_function)(samples)
        # Squeeze the energy_history second dim. so to enable the diff computation across the sample dimension.
        energy_history = jnp.squeeze(energy_history, axis=1)
        jax.debug.print("Energy history: {x}", x=energy_history)
        energy_diffs = jnp.diff(energy_history)
        numerator = jnp.mean(jnp.square(energy_diffs))
        jax.debug.print("Energy diffs: {x}", x=energy_diffs)
        denominator = jnp.var(energy_history, axis=-1)
        jax.debug.print("Energy history: {x}", x=energy_history)
        bfmi_per_chain = numerator / denominator
        jax.debug.print("BFMI per chain: {x}", x=bfmi_per_chain)
        return jnp.min(bfmi_per_chain)
    
    def compute_normalized_rank_split(self, samples: Array) -> float:
        """
        Computes the rank-normalized split diagnostic.
        """
        def _attribute_normal_ranks(samples: Array) -> Array:
            original_shape = samples.shape
            flattened_chains = samples.ravel()
            # Compute the ranks of each element in the flattened array.
            # Jax will take care of communicating across devices to compute the ranks globally across all chains.
            ranks = jnp.argsort(jnp.argsort(flattened_chains))
            # Map to 0-1 with Blom's formula
            n_total = len(flattened_chains)
            normalised_ranks = (ranks - 0.375) / (n_total + 0.25)
            # Transform to z-scores 
            z_scores = ndtri(normalised_ranks)
            return z_scores.reshape(original_shape)
        
        def _calculate_r_hat_anova(samples: Array) -> Tuple[Array, float, float]:
            z_scores = _attribute_normal_ranks(samples)
            # Reshape again to separate the chains and the samples within each chain.
            # The shape is now (num_chains, num_samples, sample_dim).
            z_scores = jnp.reshape(z_scores, (self.parallel_chains, -1, self.shape))
            jax.debug.print("Z-scores shape: {x}", x=z_scores.shape)
            half_len = z_scores.shape[1] // 2
            split_chains = jnp.concatenate([z_scores[:, :half_len, :], z_scores[:, half_len:, :]], axis=0)
            num_chains, num_samples, sample_dim = split_chains.shape
            jax.debug.print("Shape of split chains: {x}", x=split_chains.shape)
                
            chains_means = jnp.nanmean(split_chains, axis=1)
            chains_vars = jnp.nanvar(split_chains, axis=1, ddof=1)

            # * Compute the inter-chains variance
            inter_chain_variance = jnp.nanvar(chains_means, axis=0, ddof=1)
            # * Compute the intra-chains variance
            intra_chain_variance = jnp.nanmean(chains_vars, axis=0)
            # * Compute the estimated marginal posterior variance
            var_plus = ((num_samples - 1) / num_samples) * inter_chain_variance + (1 / num_samples) * intra_chain_variance

            # * Compute the potential scale reduction factor
            jax.debug.print("var_plus: {x}", x=var_plus)
            jax.debug.print("intra_chain_variance: {x}", x=intra_chain_variance)
            jax.debug.print("inter_chain_variance: {x}", x=inter_chain_variance)
            r_hat = jnp.sqrt(var_plus / jnp.maximum(intra_chain_variance, 1e-12))
            # * Instead of drowning in high dimensional R_hat, we can use a conservative summary statistic, by taking the maximum plus the mean.
            r_hat_max = jnp.max(r_hat)
            r_hat_mean = jnp.mean(r_hat)
            jax.debug.print("Normalized rank split: {x}", x=r_hat_max)
            return {"r_hat": r_hat, "r_hat_max": r_hat_max}
    
        r_hat_stats = _calculate_r_hat_anova(samples)

        # Folded version of R_hat. Fold samples by taking the absolute distance to the median, to check for convergence issues in the tails of the distribution.
        median = jnp.median(samples)
        folded_samples = jnp.abs(samples - median)
        r_hat_folded_stats = _calculate_r_hat_anova(folded_samples)

        return r_hat_stats | {"r_hat_folded": r_hat_folded_stats["r_hat"], "r_hat_folded_max": r_hat_folded_stats["r_hat_max"]}
    
    @jax.jit(static_argnums=(0,))
    def compute_mcmc_diagnostics(self, samples: Array) -> dict:
        samples = jnp.log(samples)
        rhat_stats = self.compute_normalized_rank_split(samples)
        ebmfi = self.compute_ebfmi(samples)
        return rhat_stats | {"ebfmi": ebmfi}
    
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
        self, mu_0, nu_0, lambda_1, lambda_2, lambda_I_1, lambda_I_2, cost_fn, epsilon
    ):
        self.mu_0 = jnp.asarray(mu_0)
        self.nu_0 = jnp.asarray(nu_0)
        self.cost_fn = cost_fn
        self.epsilon = epsilon
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_I_1 = lambda_I_1
        self.lambda_I_2 = lambda_I_2
        self.II = len(self.mu_0)
        self.JJ = len(self.nu_0)
        self.pi_I = jnp.exp(- self.cost_fn / self.epsilon)
        self.pi_I = jnp.reshape(self.pi_I, (1, -1))
        self.init_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

    def hyperprior_log_prob_fun(self, pi: Array) -> float:
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

    @staticmethod
    def safe_log(p: Array, epsilon=1e-15)->Array:
        return jnp.log(p + epsilon)

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

    def hyperprior_score_fun(self, pi: Array) -> Array:
        """Evaluates the score function of the HFPD-OT hyperprior at $\pi$, a vector of size (1, II x JJ)

        Args:
            pi (NDArray): flattened probability vector of shape (1, nxm)

        Returns:
            _type_: _description_
        """
        assert pi.shape[1] == self.II * self.JJ
        mat = pi.reshape((self.II, self.JJ))
        mu = jnp.sum(mat, axis=1)
        nu = jnp.sum(mat, axis=0)
        logger.info(f"Shape of the first marginal: {mu.shape}")
        logger.info(f"Shape of the second marginal: {nu.shape}")
        assert mu.shape == self.mu_0.shape
        assert nu.shape == self.nu_0.shape
        logger.info("Initiating the gradient computation")

        def _compute_single_gradient(idx):
            """
            Computes the gradient of the log-hyperprior with respect to a single entry of the flattened transport plan vector pi.
            Note that the gradient is computed with respect to log_pi, the log of pi, to ensure the positivity constraint on pi.
            """
            first_grad_term = - HFPDOTHyperprior.safe_log(pi[1, idx]) + HFPDOTHyperprior.safe_log(self.pi_I[1, idx]) - 1
            # jax.debug.print("Grad first_grad_term: {x}", x=first_grad_term)
            second_grad_term = - (self.lambda_1 + self.lambda_I_1) * (
                HFPDOTHyperprior.safe_log(mu[idx // self.JJ]) - HFPDOTHyperprior.safe_log(self.mu_0[idx // self.JJ]) + 1
            )
            # jax.debug.print("Grad second_grad_term: {x}", x=second_grad_term)
            third_grad_term = - (self.lambda_2 + self.lambda_I_2) * (
                HFPDOTHyperprior.safe_log(nu[idx % self.JJ]) - HFPDOTHyperprior.safe_log(self.nu_0[idx % self.JJ]) + 1
            )
            # jax.debug.print("Grad third_grad_term: {x}", x=third_grad_term)
            grad = first_grad_term + second_grad_term + third_grad_term + 1
            # * Zero out gradients that have gone off the rails
            grad = jnp.where(jnp.isfinite(grad), grad, 0)
            return grad
        
        # Estimate gradients in parallel across all cores.
        p_indices = jnp.arange(self.II * self.JJ)
        gradients = jax.vmap(_compute_single_gradient)(p_indices)
        # Clip the gradients' norm to avoid Nans
        threshold = 1
        grad_norm = jnp.linalg.norm(gradients)
        # jax.debug.print("grads before norm: {x}", x=gradients)
        # jax.debug.print("grad_norm: {x}", x=grad_norm)
        gradients = jnp.where(grad_norm > threshold, gradients / grad_norm * threshold, gradients)
        # jax.debug.print("grads after norm: {x}", x=gradients)
        logger.info("Completed the gradient computation")  

        return gradients
