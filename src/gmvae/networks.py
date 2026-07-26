import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from gmvae.layers import Gaussian, GaussianOuts
from gmvae.losses import LossFunctions

_LOG_2PI = math.log(2.0 * math.pi)

@dataclass(frozen=True)
class GMVAEOutput:
    """The single VaDE contract the rest of the pipeline reads. The posterior fields
    (mean_latent, var_latent, prob_cat) are exactly wadd_dim_reduction.CellPosterior's,
    so the embedder wrapper maps a trained model onto CellPosterior with no reshaping."""
    # Posterior q(z, c | x)
    mean_latent: torch.Tensor      # mu~(x)                         [B, D]
    logvar_latent: torch.Tensor    # log sigma~^2(x)                [B, D]
    latent_sample: torch.Tensor    # z ~ q(z|x)                     [B, D]
    log_gamma_c: torch.Tensor      # log responsibilities log gamma [B, K]
    # Generative p(x | z): decoder log-rates over genes
    x_recon: torch.Tensor          #                                [B, D]
    # ELBO terms (all scalars: summed over their own dims, mean over cells)
    recon_loss: torch.Tensor
    gaussian_loss: torch.Tensor
    categorical_loss: torch.Tensor
    total_loss: torch.Tensor

    @property
    def var_latent(self) -> torch.Tensor:
        return torch.exp(self.logvar_latent)

    @property
    def prob_cat(self) -> torch.Tensor:
        """The soft membership gamma (responsibilities), normalised over components."""
        return torch.exp(self.log_gamma_c)

@dataclass(frozen=True)
class InferenceOuts:
    mean_latent: torch.Tensor
    logvar_latent: torch.Tensor
    latent_samples: torch.Tensor
    log_gamma_c: torch.Tensor

class ClusterPrior(nn.Module):
    # Prior p(z|c) := N(μ_c, diag σ_c²), a lookup table of learnable parameters indexed
    # by cluster identity, with no dependence on x.
    def __init__(self, K, latent_dim):
        super().__init__()
        self.pi_logits = nn.Parameter(torch.randn(K,) * 0.5)                 # Learnable logits
        self.mu_c = nn.Parameter(torch.randn(K, latent_dim) * 0.5)           # Learnable mu_p
        self.logvar_c = nn.Parameter(torch.zeros(K, latent_dim))             # Learnable logvar_p

    def forward(self):
        return self.pi_logits, self.mu_c, self.logvar_c   # [K,], [K, d], [K, d]
    
class InferenceNet(nn.Module):
    def __init__(self, x_dim: int,
                hidden_dim: int,
                latent_dim: int,
                cluster_prior: ClusterPrior):
        """Encoder for q(z|x). No c-conditioning (VaDE): one Gaussian per cell; the cluster
        responsibilities gamma_c are derived from the shared cluster_prior given z.

        Args:
            x_dim: dim. of the input (gene) tensor
            hidden_dim: dim. of the hidden inference layers
            latent_dim: latent dim.
            cluster_prior: the shared GMM prior p(z|c) used to compute responsibilities
        """
        super(InferenceNet, self).__init__()
        self.cluster_prior = cluster_prior

        # Define the topology.
        # q(z|x), no dependence on y  --> VaDE
        self.inference_qzx = torch.nn.ModuleList(
            [
                nn.Linear(x_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                Gaussian(hidden_dim, latent_dim)
            ]
        )
    def q_zx(self, x) -> GaussianOuts:
        for _, layer in enumerate(self.inference_qzx):
            # Last layer is a unimodal Gaussian
            x = layer(x)
        return x
        
    def responsibilities(self, z):
        # q(c|z) = gamma_c, the exact posterior of c under the GMM prior given z:
        #   gamma_c = softmax_c[ log pi_c + log N(z; mu_c, sigma_c^2) ].
        # Returned in log-space (log_softmax) so the ELBO terms stay numerically stable; the
        # consumers take exp() where they need the probabilities. Computed in torch throughout
        # so the gradient reaches the encoder and the GMM prior (mu_c, logvar_c, pi_logits).
        log_pi = F.log_softmax(self.cluster_prior.pi_logits, dim=-1)   # [K]
        mu_c = self.cluster_prior.mu_c                                 # [K, d]
        logvar_c = self.cluster_prior.logvar_c                         # [K, d]
        # log N(z_b; mu_c, diag sigma_c^2) for every (cell b, component c) -> [B, K].
        # z[:, None, :] - mu_c broadcasts to [B, K, d]; the sum contracts the latent dim.
        diff = z[:, None, :] - mu_c                                    # [B, K, d]
        log_norm = -0.5 * torch.sum(
            _LOG_2PI + logvar_c + diff.pow(2) / torch.exp(logvar_c), dim=-1
        )                                                             # [B, K]
        return F.log_softmax(log_pi + log_norm, dim=-1)               # [B, K]
        
    def forward(self, x) -> InferenceOuts:
        # First, q(z|x)
        gaussian_outs = self.q_zx(x,)
        # Then the prior clusters params
        log_gamma_c = self.responsibilities(gaussian_outs.latent_samples)
        return InferenceOuts(mean_latent=gaussian_outs.mu, 
                            logvar_latent=gaussian_outs.logvar, 
                            latent_samples=gaussian_outs.latent_samples,
                            log_gamma_c=log_gamma_c,)

class GenerativeNet(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim, cluster_prior: ClusterPrior):
        super(GenerativeNet, self).__init__()

        # Read the Gaussian params from the same cluster_prior --> consistent latent model
        self.mu_c = cluster_prior.mu_c
        self.logvar_c = cluster_prior.logvar_c

        # log-dispersion is a learnable tensor, one entry per gene.
        self.log_dispersion = nn.Parameter(torch.zeros(x_dim))

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        ])

    @property
    def dispersion(self):
        return F.softplus(self.log_dispersion)

    def p_xz(self, z):
        output = z
        for layer in self.generative_pxz:
            output = layer(output)
        return output

    def forward(self, z) -> torch.Tensor:
        # Decoder log-rates over genes: log m(z), the NB mean parameter. [B, D]
        return self.p_xz(z)

class GMVAENet(nn.Module):

    def __init__(self, x_dim, num_clusters, latent_dim, hidden_dim, beta=1.0):
        super(GMVAENet, self).__init__()
        self.cluster_prior = ClusterPrior(num_clusters, latent_dim)
        self.inference_net = InferenceNet(x_dim, hidden_dim, latent_dim, self.cluster_prior)
        self.generative_net = GenerativeNet(x_dim, hidden_dim, latent_dim, self.cluster_prior)
        self.num_clusters = num_clusters
        self.mu_c = self.cluster_prior.mu_c
        self.logvar_c = self.cluster_prior.logvar_c
        self.pi_logits = self.cluster_prior.pi_logits
        # Explicit KL weight (spec decision 3): minimise -(1) - beta*[(2)+(3)+(4)]. Never an implicit
        # reduction factor. beta=1 is the plain ELBO.
        self.beta = beta
        self.losses = LossFunctions()

        # Weights initialisation with He init
        # TODO: is this the right init here?
        for m in self.modules():
            if type(m) is nn.Linear:
                nn.init.kaiming_normal_(m.weight)
        
    def forward(self, x):
        inf = self.inference_net(x)
        # Decoder-only generative net: p(x|z) reconstructs from the inference sample z ~ q(z|x).
        x_recon = self.generative_net(inf.latent_samples)

        # (1) reconstruction; (2)+(3) gamma-weighted latent KL to the GMM prior; (4) KL(gamma||pi).
        recon_loss = self.losses.reconstruction_loss(x, x_recon, self.generative_net.dispersion)
        gaussian_loss = self.losses.gaussian_loss(inf.mean_latent, inf.logvar_latent,
                                                  inf.log_gamma_c, self.mu_c, self.logvar_c)
        categorical_loss = self.losses.categorical_loss(self.pi_logits, inf.log_gamma_c)
        total_loss = recon_loss + self.beta * (gaussian_loss + categorical_loss)

        return GMVAEOutput(
            mean_latent=inf.mean_latent,
            logvar_latent=inf.logvar_latent,
            latent_sample=inf.latent_samples,
            log_gamma_c=inf.log_gamma_c,
            x_recon=x_recon,
            recon_loss=recon_loss,
            gaussian_loss=gaussian_loss,
            categorical_loss=categorical_loss,
            total_loss=total_loss,
        )

    # -- Stage A: plain-VAE pretraining (build a latent before the GMM is seeded) -------------------

    def pretrain_parameters(self):
        """The Stage A trainable set: encoder + decoder + per-gene dispersion, and ONLY those.

        Explicitly EXCLUDES the GMM prior (mu_c, logvar_c, pi_logits) so it is impossible to update
        the prior during pretraining -- the prior is set later by SEED, not learned in Stage A.
        Enumerated from the concrete sub-modules rather than filtering ``parameters()``, because the
        shared ``cluster_prior`` is a registered submodule of both nets and would otherwise leak in.
        """
        yield from self.inference_net.inference_qzx.parameters()   # encoder q(z|x)
        yield from self.generative_net.generative_pxz.parameters()  # decoder p(x|z)
        yield self.generative_net.log_dispersion                    # per-gene NB dispersion

    def pretrain_loss(self, x, beta=1.0):
        """Stage A objective: NB reconstruction + standard-normal KL (a plain VAE).

        Builds a structured latent by training encoder + decoder (+ dispersion) BEFORE the GMM prior
        is seeded; the mixture is not consulted here (no responsibilities), and the prior is N(0, I)
        rather than the GMM. Returns ``(total, recon, kl)`` -- watching recon vs KL is the main
        pretraining diagnostic. Decodes the sampled z (a training pass wants the reparam noise).
        """
        gauss = self.inference_net.q_zx(x)                          # mu, logvar, sampled z
        x_recon = self.generative_net(gauss.latent_samples)
        recon = self.losses.reconstruction_loss(x, x_recon, self.generative_net.dispersion)
        zeros = torch.zeros_like(gauss.mu)
        kl = self.losses.gaussian_kl_diag(gauss.mu, gauss.logvar, zeros, zeros).mean()  # KL(q||N(0,I))
        return recon + beta * kl, recon, kl
