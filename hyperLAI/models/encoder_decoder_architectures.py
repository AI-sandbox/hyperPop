import torch
from torch import nn
from models.hyperbolic_hc_loss import HyperbolicHCLoss
from pvae.ops.manifold_layers import GeodesicLayer

class fc_wrapped_encoder(nn.Module):
    def __init__(self, manifold, input_size, num_encoder_int_layers, encoder_int_layer_sizes, encoder_dropout_vals,
                                  embedding_size):
        
        super().__init__()  
        self.manifold = manifold
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.fc_layers = nn.ModuleList()
        for layer in range(num_encoder_int_layers):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=self.input_size if layer == 0 else encoder_int_layer_sizes[layer - 1],
                        out_features=encoder_int_layer_sizes[layer]
                    ),
                    nn.BatchNorm1d(num_features=encoder_int_layer_sizes[layer]),
                    nn.ReLU(),
                    nn.Dropout(encoder_dropout_vals[layer])
                )
            )
                
        last_layer_input = encoder_int_layer_sizes[-1] if num_encoder_int_layers > 0 else self.input_size
        
        self.final_layer_mean = nn.Linear(in_features=last_layer_input, out_features=self.embedding_size)    
        self.final_layer_std = nn.Linear(in_features=last_layer_input, out_features=self.embedding_size)

    def forward(self, snp_data):
        for fc_unit in self.fc_layers:
            snp_data = fc_unit(snp_data)
        loc = self.final_layer_mean(snp_data)
        loc = self.manifold.expmap0(loc)
        scale = nn.Softplus()(self.final_layer_std(snp_data)) + 1e-5
        return loc, scale

class fc_wrapped_decoder(nn.Module):
    def __init__(self, manifold, input_size, num_decoder_int_layers, decoder_int_layer_sizes, decoder_dropout_vals,
                                  embedding_size):
        
        super().__init__()
        self.manifold = manifold
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.fc_layers = nn.ModuleList()
        for layer in range(num_decoder_int_layers):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=self.embedding_size if layer == 0 else decoder_int_layer_sizes[layer - 1],
                        out_features=decoder_int_layer_sizes[layer]
                    ),
                    nn.BatchNorm1d(num_features=decoder_int_layer_sizes[layer]),
                    nn.ReLU(),
                    nn.Dropout(decoder_dropout_vals[layer])
                )
            )
                               
        last_layer_input = decoder_int_layer_sizes[-1] if num_decoder_int_layers > 0 else self.embedding_size
        
        self.final_layer = nn.Linear(in_features=last_layer_input, out_features=self.input_size)       

    def forward(self, embeddings):
        embeddings = self.manifold.logmap0(embeddings)
        for fc_unit in self.fc_layers:
            embeddings = fc_unit(embeddings)
        recon_data = nn.Sigmoid()(self.final_layer(embeddings))
        return recon_data

class fc_geodesic_decoder(nn.Module):
    def __init__(self, manifold, input_size, num_decoder_int_layers, decoder_int_layer_sizes, decoder_dropout_vals,
                                  embedding_size):
        
        super().__init__()
        self.manifold = manifold
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.geodesiclayer = nn.Sequential(GeodesicLayer(self.manifold.coord_dim, embedding_size, self.manifold),
                                           nn.ReLU())
        self.fc_layers = nn.ModuleList()
        for layer in range(num_decoder_int_layers):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=self.embedding_size if layer == 0 else decoder_int_layer_sizes[layer - 1],
                        out_features=decoder_int_layer_sizes[layer]
                    ),
                    nn.BatchNorm1d(num_features=decoder_int_layer_sizes[layer]),
                    nn.ReLU(),
                    nn.Dropout(decoder_dropout_vals[layer])
                )
            )
            
        last_layer_input = decoder_int_layer_sizes[-1] if num_decoder_int_layers > 0 else self.embedding_size
        
        self.final_layer = nn.Linear(in_features=decoder_int_layer_sizes[-1], out_features=self.input_size)       

    def forward(self, embeddings):
        # embeddings = self.geodesiclayer(embeddings.unsqueeze(-2)).squeeze()
        embeddings = self.geodesiclayer(embeddings.unsqueeze(dim=0)).squeeze()
        for fc_unit in self.fc_layers:
            embeddings = fc_unit(embeddings)
        recon_data = nn.Sigmoid()(self.final_layer(embeddings))
        return recon_data
