import sys
sys.path.append("../")
import torch
from torch import nn
from models.hyperbolic_hc_loss import HyperbolicHCLoss

class fc_model(nn.Module):
    def __init__(self, input_size, num_int_layers, int_layer_sizes, embedding_size, dropout_vals,
                temperature=0.05, init_size=1e-3, min_scale=1e-2, max_scale=1. - 1e-3):
        super().__init__()
        assert len(int_layer_sizes) == num_int_layers and len(dropout_vals) == num_int_layers
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.fc_layers = nn.ModuleList()
        for layer in range(num_int_layers):
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=self.input_size if layer == 0 else int_layer_sizes[layer - 1],
                        out_features=int_layer_sizes[layer]
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout_vals[layer])
                )
            )
                
        self.final_layer = nn.Linear(in_features=int_layer_sizes[-1], out_features=embedding_size)
        self.HypLoss = HyperbolicHCLoss(temperature, init_size, min_scale, max_scale)
        
    def forward(self, snp_data):
        for fc_unit in self.fc_layers:
            snp_data = fc_unit(snp_data)
        embeddings = self.final_layer(snp_data)
        return embeddings
    
    def calculate_loss(self, embeddings, triple_ids, similarities):
        return self.HypLoss(embeddings, triple_ids, similarities)
                
                    