import argparse
import torch
import allel
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../../libraries/")
from HypHC.optim.radam import RAdam
from HypHC.utils.poincare import project
from HypHC.utils.visualization import plot_tree_from_leaves
from HypHC.utils.linkage import nn_merge_uf_fast_np, sl_from_embeddings
from HypHC.utils.metrics import dasgupta_cost
sys.path.append("../../libraries/pvae/")
from pvae.manifolds.poincareball import PoincareBall
from pvae.distributions.wrapped_normal import WrappedNormal
import torch
from torch.distributions.normal import Normal
from torch.utils import data
sys.path.append("../hyperLAI")
from models.encoder_decoder_architectures import *
from models.vae_model import vae_model
from models.fc_model import fc_model


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SimpleSNPDataset(data.Dataset):
    '''
    Simple PyTorch dataset class to use in this script
    '''
    def __init__(self, snp_data):
        self.snp_data = snp_data
    def __len__(self):
        return len(self.snp_data)
    def __getitem__(self, index):
        return torch.tensor(self.snp_data[index])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_vcf", type=str)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_type", type=str, choices=["HypVAE", "HypMLP"])
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()

def load_data(input_vcf, label_file):
    '''
    Loads the SNP data and labels into formats that can be used in the later functions
    '''
    #Load vcf file from skallel 
    vcf_snp_data = allel.read_vcf(input_vcf)['calldata/GT']
    snp_matrix = vcf_snp_data.reshape(vcf_snp_data.shape[0], 
                                        vcf_snp_data.shape[1] * vcf_snp_data.shape[2]).T
    snp_torch_dataset = SimpleSNPDataset(snp_matrix)
    snp_loader = data.DataLoader(snp_torch_dataset, batch_size=64)
    
    #Get labels and make dict from them
    label_list = []
    for label in open(label_file, "r"):
        label_list.append(label.strip())
        label_list.append(label.strip())
    return snp_loader, label_list

def make_pairwise_similarities(data, sim_func):
    '''
    Creates a similarity matrix from the SNP data using the specified similarity function
    This is a numpy version of the function used in training
    '''
    sim_matrix = np.zeros((data.shape[0], data.shape[0]))
    #Fill in matrix
    for ind in range(data.shape[0]):
        for ind2 in range(data.shape[0]):
            sim_matrix[ind][ind2] = sim_func(data[ind], data[ind2])
    #Divide by maximum for normalization
    sim_matrix /= np.amax(sim_matrix)
    return sim_matrix

def load_model(model_dir, model_type):
    '''
    Loads specified model from model_dir
    '''
    if model_type == "HypVAE":
        manifold = PoincareBall(2)
        enc_type, dec_type  = fc_wrapped_encoder, fc_wrapped_decoder
        encoder = enc_type(manifold, 500000, 3, [300,200,100], [0.2,0.2,0.2], 2)
        decoder = dec_type(manifold, 500000, 3, [100,200,300], [0.2,0.2,0.2], 2)
        model = vae_model(encoder, decoder, manifold, WrappedNormal, WrappedNormal, 0.0, 1.0, 1e-6, 1e-3, 1e-2, 0.999)
        model_info = torch.load("%s/vae_model.pt"%(model_dir))
        model.load_state_dict(model_info["model_state"])
    else:
        model = fc_model(500000, 3, [300,200,100], 2, [0.2, 0.2, 0.2], 0.0001, 1e-3, 1e-2, 0.999)
        model_info = torch.load("%s/fc_model.pt"%(model_dir))
        model.load_state_dict(model_info["model_state"])
    return model


def predict(loader, model, model_type, output_dir):
    '''
    Makes predictions using model and saves to file
    '''
    snps, embeddings = [], []
    with torch.no_grad():
        for i, snp_data in enumerate(loader):
            if model_type == "HypVAE":
                embs = model.embed(snp_data.float().to(device))
            else:
                embs = model(snp_data.float().to(device))
            embeddings.append(embs.cpu())
            snps.append(snp_data.cpu())
    snps = torch.cat(snps).numpy()
    embeddings = torch.cat(embeddings).numpy()
    np.save(output_dir + "predicted_embeddings.npy", embeddings)
    return snps, embeddings

def plot_weights(embeddings, labels, output_dir):
    '''
    Plots the predicted embeddings
    '''
    sns.set_style('white')
    scplot = sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], hue=labels)
    plt.xlabel("Embedding 1", fontsize=16)
    plt.ylabel("Embedding 2", fontsize=16)
    plt.title("Embedding Weights", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(output_dir + "embedding_plot.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    return scplot



def make_tree(model, embeddings, output_dir, model_type):
    """Build a binary tree (nx graph) from leaves' embeddings. Assume points are normalized to same radius.
        Taken from HypHC repo (https://github.com/HazyResearch/HypHC)
    
    """
    with torch.no_grad():
        if model_type == "HypVAE":
            leaves_embeddings = model.HypHCLoss.normalize_embeddings(torch.tensor(embeddings).to(device))
        else:
            leaves_embeddings = model.HypLoss.normalize_embeddings(torch.tensor(embeddings).to(device))
        leaves_embeddings = project(leaves_embeddings).cpu()
    sim_fn = lambda x, y: torch.sum(x * y, dim=-1)
    parents = nn_merge_uf_fast_np(leaves_embeddings, S=sim_fn, partition_ratio=1.2)
    tree = nx.DiGraph()
    for i, j in enumerate(parents[:-1]):
        tree.add_edge(j, i)
    edges_out = open(output_dir + "tree_edges.txt", "w")
    written = [edges_out.write(str(x[0]) + "\t" + str(x[1]) + "\n") for x in tree.edges]
    return tree

def main():
    args = parse_args()
    snp_loader, labels = load_data(args.input_vcf, args.labels)
    model = load_model(args.model_dir, args.model_type).to(device)
    snps, embeddings = predict(snp_loader, model, args.model_type, args.output_dir)
    emb_plot = plot_weights(embeddings, labels, args.output_dir)
    tree = make_tree(model, embeddings, args.output_dir, args.model_type)
    
if __name__ == "__main__":
    main()
    
    

