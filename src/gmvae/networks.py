import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from gmvae.layers import GumbelSoftmax, Gaussian

@dataclass
class GMVAEOutput:
    mean_inf: torch.Tensor
    var_inf: torch.Tensor
    mean_gen: torch.Tensor
    var_gen: torch.Tensor
    x_recon: torch.Tensor
    latent_sample: torch.Tensor
    logits: torch.Tensor
    prob_cat: torch.Tensor
    categorical: torch.Tensor
    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    gaussian_loss: torch.Tensor
    categorical_loss: torch.Tensor

class InferenceNet(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, latent_dim):
        """_summary_

        Args:
            x_dim (_type_): dim. of the input tensor
            y_dim (_type_): number of discrete categories (modes)
            hidden_dim (_type_): dim. of the hidden inference layers 
            latent_dim (_type_): latent dim.
        """
        super(InferenceNet, self).__init__()
        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList(
            [
                nn.Linear(x_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                GumbelSoftmax(hidden_dim, y_dim)
            ]
        )
        # q(z|x,y)
        self.inference_qzxy = torch.nn.ModuleList(
            [
                nn.Linear(x_dim + y_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                Gaussian(hidden_dim, latent_dim)
            ]
        )
    # q(y|x)
    def q_yx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # Last layer is a Gumbel Softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x
        
    # q(z|x,y)
    def q_zxy(self, x, y):
        concat = torch.cat((x, y), dim=-1)
        for layer in self.inference_qzxy:
            concat = layer(concat)
        return concat
        
    def forward(self, x, temperature, hard):
        # q(y|x)
        logits, prob, y = self.q_yx(x, temperature, hard)
        # q(z|x,y)
        mu, var, z = self.q_zxy(x, y)
        return {"mean_inf": mu, 
                "var_inf": var, 
                "latent_sample": z,
                "logits": logits,
                "prob_cat": prob, 
                "categorical": y}

class GenerativeNet(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim, latent_dim):
        super(GenerativeNet, self).__init__()

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, latent_dim)
        self.y_var = nn.Linear(y_dim, latent_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
            nn.ReLU()
        ])

    # p(z|y)
    def p_zy(self, y):
        mu = self.y_mu(y)
        var = F.softplus(self.y_var(y)) 
        return mu, var
    
    def p_xz(self, z):
        output = z
        for layer in self.generative_pxz:
            output = layer(output)
        return output

    def forward(self, z, y):
        mu, var = self.p_zy(y)
        x_recon = self.p_xz(z)
        output = {"mean_gen": mu, "var_gen": var, "x_recon": x_recon}
        return output

class GMVAENet(nn.Module):

    def __init__(self, x_dim, y_dim, latent_dim, hidden_dim):
        super(GMVAENet, self).__init__()
        self.inference_net = InferenceNet(x_dim, y_dim, hidden_dim, latent_dim)
        self.generative_net = GenerativeNet(x_dim, y_dim, hidden_dim, latent_dim)

        # Weights initialisation with He init
        for m in self.modules():
            if type(m) is nn.Linear:
                nn.init.kaiming_normal_(m.weight)
        
    def forward(self, x, temperature, hard):
        inference_outs = self.inference_net(x, temperature, hard)
        z, y = inference_outs["latent_sample"], inference_outs["categorical"]
        generative_outs = self.generative_net(z, y)

        # Merge outputs
        outs_all = inference_outs | generative_outs
        outs_all["total_loss"] = 0
        outs_all["recon_loss"] = 0
        outs_all["gaussian_loss"] = 0
        outs_all["categorical_loss"] = 0
        gmvae_outs = GMVAEOutput(**outs_all)
        return gmvae_outs


    
