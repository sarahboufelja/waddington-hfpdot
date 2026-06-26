import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmax(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        
    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        u = torch.rand(shape)
        if is_cuda:
            u = u.cuda()
        return -torch.log(-torch.log(u + eps) + eps)
    
    def sample_gumbel_softmax(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)
    
    def gumbel_softmax(self, logits, temperature, hard=False):
        y = self.sample_gumbel_softmax(logits, temperature)

        if not hard:
            return y
        
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # straight-through estimator (STE)
        y_hard = (y_hard - y).detach() + y
        return y_hard
    
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.out_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


class Gaussian(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, latent_dim)
        self.var = nn.Linear(in_dim, latent_dim)

    def reparametrization(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.rand_like(std)
        z = mu + noise * std
        return z
    
    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparametrization(mu, var)
        return mu, var, z
