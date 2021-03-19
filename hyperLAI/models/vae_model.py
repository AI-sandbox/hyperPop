import sys
import torch
from torch import nn
from models.hyperbolic_hc_loss import HyperbolicHCLoss

#VAE model class
#Once again inspired by models found in the pvae repo (https://github.com/emilemathieu/pvae/)

class vae_model(nn.Module):
    def __init__(self, encoder, decoder, manifold, posterior_dist, prior_dist, prior_mean=0.0, prior_std=1.0,
                 temperature=1e-3, init_size=1e-3, min_scale=1e-2, max_scale=1. - 1e-3):
        '''
        Initializes the vae model
        Arguments:
            `encoder: pytorch model. Must input and return the same values as the encoders defined in this repo
            `decoder: pytorch model. Must input and return the same values as the decoders defined in this repo
            `manifold: manifold used for the model
            `posterior_dist (PyTorch distribution-like): distribution to use for the posterior. Should be un-initialized (ie. "Normal", not "Normal(0,1)")
            `prior_dist (PyTorch distribution-like): distribution to use for the prior
            `prior_mean (float): mean of the prior distribution
            `prior_std (float): stdev of the prior distribution (or, more accurately, whatever the second parameter is)
            `temperature (float): scale factor used in Hyperbolic HC Loss calculations (see HypHC paper)
            `init_size, min_scale, max_scale: further parameters for Hyperbolic HC
        '''
        super().__init__()        
        self.encoder = encoder
        self.decoder = decoder
        self.HypHCLoss = HyperbolicHCLoss(temperature, init_size, min_scale, max_scale)
        self.qz_x = posterior_dist
        self.p_z = prior_dist
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.manifold = manifold

    def forward(self, snp_data):
        '''
        Takes in input data and reconstructs it. 
        Returns:
            mean and stdev of the posterior distribution
            data embeddings
            fitted posterior distributioni
            reconstructed data
        This function is intended to be used in training 
        Using "embed" or "generate" will probably be easier to actually produce embeddings/reconstructions
        '''
        loc, scale = self.encoder(snp_data)
        qzx_fitted = self.qz_x(loc, scale, self.manifold)
        z = qzx_fitted.rsample(torch.Size([1])).squeeze()
        reconstructions = self.decoder(z)
        return loc, scale, z, qzx_fitted, reconstructions
    
    def embed(self, snp_data):
        '''
        Takes in input data and returns the embedding of the data
        '''
        loc, scale = self.encoder(snp_data)
        qzx_fitted = self.qz_x(loc, scale, self.manifold)
        z = qzx_fitted.rsample(torch.Size([1])).squeeze()
        return z
    
    def generate(self, embedding):
        '''
        Takes in embeddings and reconstructs the data from them
        '''
        return self.decoder(embedding)
        
    def calculate_hyphc_loss(self, embeddings, triple_ids, similarities):
        '''
        Calculates the hyperbolic hc loss (included here because there is a trainable parameter)
        Arguments:
            `embeddings (torch tensor): embeddings of data
            `triple_ids (torch tensor): sets of three indices (with indices corresponding to data points)
            `similarities (torch tensor): pairwise similarity values for each triple. See utils/model_utils.py to see how thesee are calculated
        '''
        return self.HypHCLoss(embeddings, triple_ids, similarities)
        
        
