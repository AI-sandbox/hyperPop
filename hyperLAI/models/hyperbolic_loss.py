import torch
from torch import nn
import torch.nn.functional as F
import sys
from HypHC.utils.lca import hyp_lca


class HyperbolicLoss(nn.Module):
    """
    Hyperbolic embedding loss function for hierarchical clustering.
    Adapted from HypHC by Hazy Research Group (https://github.com/HazyResearch/HypHC/)
    """

    def __init__(self, temperature=0.05, init_size=1e-3, min_scale=1e-2, max_scale=1. - 1e-3):
        super().__init__()
        self.temperature = temperature
        self.scale = nn.Parameter(torch.Tensor([init_size]), requires_grad=True)
        self.init_size = init_size
        self.min_scale = min_scale
        self.max_scale = max_scale

    def anneal_temperature(self, anneal_factor):
        """

        @param anneal_factor: scalar for temperature decay
        @type anneal_factor: float
        """
        self.temperature *= anneal_factor

    def normalize_embeddings(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = self.min_scale
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * self.scale.clamp_min(min_scale).clamp_max(max_scale)

    def forward(self, embeddings, triple_ids, similarities):
        """Computes the HypHC loss.
        Args:
            triple_ids: B x 3 tensor with triple ids
            similarities: B x 3 tensor with pairwise similarities for triples 
                          [s12, s13, s23]
        """
        e1 = embeddings[triple_ids[:, 0]]
        e2 = embeddings[triple_ids[:, 1]]
        e3 = embeddings[triple_ids[:, 2]]
        e1 = self.normalize_embeddings(e1)
        e2 = self.normalize_embeddings(e2)
        e3 = self.normalize_embeddings(e3)
        d_12 = hyp_lca(e1, e2, return_coord=False)
        d_13 = hyp_lca(e1, e3, return_coord=False)
        d_23 = hyp_lca(e2, e3, return_coord=False)
        lca_norm = torch.cat([d_12, d_13, d_23], dim=-1)
        weights = torch.softmax(lca_norm / self.temperature, dim=-1)
        w_ord = torch.sum(similarities * weights, dim=-1, keepdim=True)
        total = torch.sum(similarities, dim=-1, keepdim=True) - w_ord
        return torch.mean(total)