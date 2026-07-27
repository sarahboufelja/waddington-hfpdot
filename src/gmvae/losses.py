import torch
import torch.nn.functional as F


class LossFunctions:
    """
    """
    @staticmethod
    def NBNLLLoss(prediction, real, dispersion):
        # The model outputs the average number of successes in a sequence of independent
        # Bernoulli experiments before a given dispersion is observed.
        
        # Compute the logits
        logits = prediction - torch.log(dispersion)

        # Instantiate the NB dist
        nb_dist = torch.distributions.NegativeBinomial(total_count=dispersion, logits=logits)

        # Compute the negative log-likelihood as measure of goodness of fit
        return - nb_dist.log_prob(real)

    def reconstruction_loss(self, real: torch.Tensor, prediction: torch.Tensor, dispersion: torch.Tensor)->torch.Tensor:
        """
        Reconstruction loss: −Σ_g log NegativeBinomial(x_g | log_mu_g, log_dispersion).
        This is a sum over the D genes — a joint log-likelihood
        of D independent variables — then a mean over the batch
        (Monte-Carlo over datapoints).
        """
        nbll_loss = LossFunctions.NBNLLLoss(prediction, real, dispersion)
        # Sum over the dimensions and average over the batch.
        return torch.mean(torch.sum(nbll_loss, axis=-1))
    
    def gaussian_kl_diag(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        Latent KL between the two latent diagonal Gaussians given a class k --> has the following closed form:
        KL = ½ Σ_d [ σ²_q/σ²_p + (μ_p−μ_q)²/σ²_p − 1 + log(σ²_p/σ²_q) ]
        """
        var_q, var_p = torch.exp(logvar_q), torch.exp(logvar_p)
        kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0)
        return kl.sum(-1)   # Sum over the d-dim --> [B,]

    def gaussian_loss(self, mu_q, logvar_q, log_gamma_c, mu_p, logvar_p):
        # mu_q, logvar_q: [B, d] --> q(z|x); log_gamma_c: [B, K] --> log responsibilities
        # ELBO terms (2)+(3): sum_c gamma_c * KL(q(z|x) || N(mu_c, sigma_c^2)).
        kl_per_cluster = self.gaussian_kl_diag(
            mu_q[:, None, :], logvar_q[:, None, :],  # [B, 1, d]
            mu_p[None, :, :], logvar_p[None, :, :]   # [1, K, d]
        )  # --> kl_per_cluster: [B, K]
        gamma_c = torch.exp(log_gamma_c)   # [B, K]
        return torch.mean((gamma_c * kl_per_cluster).sum(dim=-1)) # Sum over all clusters --> [B,] then mean over cells --> () 

    def categorical_loss(self, pi_logits, log_gamma_c):
        r"""Computes the categorical KL: KL(\gamma || \pi)
        """
        gamma = torch.exp(log_gamma_c)                    # log_gamma_c is already log-normalised
        log_pi = F.log_softmax(pi_logits, dim=-1)
        return torch.mean(torch.sum(gamma * (log_gamma_c - log_pi), axis=-1))

    def anchor_loss(self, log_gamma_c, targets, class_weights):
        r"""Stage B supervised anchor: soft multi-label cross-entropy on the labelled cells,
        class-balanced. Pulls each labelled cell's responsibilities toward its (soft) label target,
        which keeps the GMM components pinned to the populations during joint training.

            L = mean_{i labelled} [ - sum_c w_c * t_{i,c} * log gamma_{i,c} ]

        - ``log_gamma_c`` [B, K]: log responsibilities from the forward pass.
        - ``targets`` [B, K]: soft label targets, renormalised to a distribution for labelled cells
          and all-zero for unlabelled cells (which therefore contribute nothing).
        - ``class_weights`` [K]: per-population balance weights ``w_c``, folded into the class sum so
          an ambiguous (multi-membership) cell weights each of its populations by that population's
          own ``w_c`` -- no single per-cell class is assumed.

        The mean is over the labelled cells in the batch (not the whole batch), so ``lambda_sup``
        trades per-labelled-cell supervision against the per-cell ELBO at a scale that does not drift
        with how many labelled cells a random batch happens to contain.
        """
        labelled = targets.sum(-1) > 0                     # [B]; unlabelled rows are all-zero
        if not bool(labelled.any()):
            return log_gamma_c.new_zeros(())               # no labelled cells in this batch -> 0
        per_cell = -(class_weights * targets * log_gamma_c).sum(-1)   # [B]
        return per_cell[labelled].mean()                   # mean over labelled-in-batch
