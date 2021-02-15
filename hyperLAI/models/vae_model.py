import sys
sys.path.append("../")
import torch
from torch import nn
from models.hyperbolic_hc_loss import HyperbolicHCLoss

class vae_model(nn.Module):
    def __init__(self, input_size, num_encoder_int_layers, encoder_int_layer_sizes, encoder_dropout_vals, embedding_size,
                 num_decoder_int_layers, decoder_int_layer_sizes, decoder_dropout_vals,
                 manifold, posterior_dist, prior_dist, prior_mean=0.0, prior_std=1.0,
                 temperature=1e-3, init_size=1e-3, min_scale=1e-2, max_scale=1. - 1e-3):
        super().__init__()        
        self.encoder = fc_encoder(manifold, input_size, num_encoder_int_layers, encoder_int_layer_sizes, encoder_dropout_vals,
                                  embedding_size)
        self.decoder = fc_decoder(manifold, input_size, num_decoder_int_layers, decoder_int_layer_sizes, decoder_dropout_vals,
                                  embedding_size)

        self.HypHCLoss = HyperbolicHCLoss(temperature, init_size, min_scale, max_scale)
        self.qz_x = posterior_dist
        self.p_z = prior_dist
        self.px_z = Normal
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.manifold = manifold

    def forward(self, snp_data):
        loc, scale = self.encoder(snp_data)
        qzx_fitted = self.qz_x(loc, scale, self.manifold)
        z = qzx_fitted.rsample(torch.Size([1])).squeeze()
        reconstructions = self.decoder(z)
        return loc, scale, z, qzx_fitted, reconstructions
    
    def embed(self, snp_data):
        loc, scale = self.encoder(snp_data)
        qzx_fitted = self.qz_x(loc, scale, self.manifold)
        z = qzx_fitted.rsample(torch.Size([1])).squeeze()
        return z
    
    def generate(self, embedding):
        return self.decoder(embedding)
        
    def calculate_hyphc_loss(self, embeddings, triple_ids, similarities):
        return self.HypHCLoss(embeddings, triple_ids, similarities)
        
        
