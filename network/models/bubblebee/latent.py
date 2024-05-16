# import libraries
import json
import torch
import torch.nn as nn


class Gaussian(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, mu_node, logvar_node, mu_edge, logvar_edge):

        z_node = self.sampling(mu_node, logvar_node)
        z_edge = self.sampling(mu_edge, logvar_edge)

        return z_node, z_edge

    def sampling(self, mu, logvar):
        # reparameterization
        std = torch.exp(0.5 * logvar).clamp(-3, 3)
        # standard sampling
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z
