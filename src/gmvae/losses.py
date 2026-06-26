
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LossFunctions:

    # def reconstruction_loss(self, real: torch.Tensor, prediction: torch.Tensor)->torch.Tensor:
    #     sq_err = (real - prediction).pow(2)
    #     return sq_err.sum(-1).mean()
    def reconstruction_loss(self, real: torch.Tensor, prediction: torch.Tensor)->torch.Tensor:
        poisson_loss = nn.PoissonNLLLoss(log_input=False, reduction="mean")
        return poisson_loss(prediction, real)
    
    def log_normal(self, x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, eps: float = 1e-8)->torch.Tensor:
        if eps > 0.0:
            var = var + eps
        return -0.5 * torch.sum(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)
    
    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()
    
    def categorical_loss(self, logits, num_clusters):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.mean(torch.sum(log_probs * probs, dim=-1)) + np.log(num_clusters)
