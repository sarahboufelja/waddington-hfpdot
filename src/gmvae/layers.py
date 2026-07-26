import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass(frozen=True)
class GaussianOuts:
    mu: torch.Tensor
    logvar: torch.Tensor
    latent_samples: torch.Tensor

class Gaussian(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, latent_dim)
        self.var = nn.Linear(in_dim, latent_dim)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z
    
    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        latent_sample = self.reparametrization(mu, logvar)
        return GaussianOuts(mu, logvar, latent_sample)
