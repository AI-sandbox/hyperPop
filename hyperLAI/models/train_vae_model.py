import sys
import os
sys.path.append("../../../libraries/pvae/")
from pvae.manifolds.poincareball import PoincareBall
from pvae.manifolds.euclidean import Euclidean
from pvae.models.architectures import EncWrapped, DecWrapped
from pvae.distributions.wrapped_normal import WrappedNormal
from pvae.distributions.riemannian_normal import RiemannianNormal
from pvae.ops.manifold_layers import GeodesicLayer
from pvae.objectives import vae_objective
from torch.distributions.normal import Normal
sys.path.append("../../../libraries/")
from HypHC.optim.radam import RAdam
from HypHC.utils.poincare import project
from HypHC.utils.visualization import plot_tree_from_leaves
from HypHC.utils.linkage import nn_merge_uf_fast_np, sl_from_embeddings
sys.path.append("../")
import math
import torch
from torch import nn
import networkx as nx
from models.hyperbolic_hc_loss import HyperbolicHCLoss
from models.encoder_decoder_architectures import *
from models.vae_model import vae_model
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset
from torch.optim import Adam
from utils.sim_funcs import sim_func_dict
from utils.model_utils import *
from features.hyperLAIdataset import HyperLoader
from models.fc_model import fc_model

enc_dec_dict = {"fc_wrapped_encoder": fc_wrapped_encoder, "fc_wrapped_decoder": fc_wrapped_decoder, "fc_geodesic_decoder": fc_geodesic_decoder} 
manifold_dict = {"PoincareBall": PoincareBall, "Euclidean": Euclidean}
distribution_dict = {"WrappedNormal": WrappedNormal, "Normal": Normal, "RiemannianNormal": RiemannianNormal}

def compute_total_loss(model, device, embeddings, 
                       reconstructions, snp_data, suppop_labels, pop_labels, kl_weight, 
                       hc_weight, recon_weight, sim_func, qzx_fitted):
    '''
    Loss function used for the variational autoencoder. Comprises KL divergence, reconstruction, and Hyperbolic HC losses. 
    Arguments:
        `model: VAE model being trained
        `device: device used for trainings
        `embeddings: embeddings produced by model
        `reconstructions: reconstructions produced by model
        `snp_data: input data
        `kl_weight (int/float): weight for KL divergence loss
        `hc_weight (int/float): weight for HC loss
        `recon_weight (int/float): weight for reconstruction loss
        `sim_func: function used to calculate pairwise similarity
        `qzx_fitted: fitted posterior distribution (pytorch-like)
    '''
    #Calculate hierarchical clustering loss
    if sim_func == sim_func_dict["ancestry_label"]:
        labels_stacked = torch.stack([suppop_labels, pop_labels], dim=1)
        triple_ids, similarities = trips_and_sims(labels_stacked, sim_func)
    else:
        triple_ids, similarities = trips_and_sims(snp_data, sim_func)
    triple_ids = triple_ids.to(device)
    similarities = similarities.float().to(device)
    hyphc_loss = model.calculate_hyphc_loss(embeddings, triple_ids, similarities)
    #Calculate KL Divergence Loss
    pz_fitted = model.p_z(torch.zeros(1, embeddings.shape[-1]).to(device) + model.prior_mean, 
                   torch.zeros(1, embeddings.shape[-1]).to(device) + model.prior_std, model.manifold)
    if (model.qz_x, model.p_z) in torch.distributions.kl._KL_REGISTRY:
        kl_div = torch.distributions.kl_divergence(qzx_fitted, pz_fitted)
    else:
        kl_div = qzx_fitted.log_prob(embeddings.unsqueeze(dim=0)) - pz_fitted.log_prob(embeddings.unsqueeze(dim=0))
    kl_div = kl_div.sum(-1).mean()
    #Calculate NLL Loss of reconstruction
    bce = nn.BCELoss(reduction="none")
    binary_loss = bce(reconstructions, snp_data).sum(-1).mean()
    total_loss = kl_weight * kl_div + hc_weight * hyphc_loss + recon_weight * binary_loss
    return total_loss, kl_weight * kl_div, hc_weight * hyphc_loss, recon_weight * binary_loss

def run_epoch(model, dloader, device, sim_func, kl_weight, hc_weight, recon_weight, optimizer=None):
    '''
    Trains or validates model for one epoch. Called in the train_model function
    `dloader: dataloader, for either training or validation
    `device: device training is occurring on
    All other parameters are same as in train_model
    '''
    total_losses, kl_losses, hyphc_losses, reconstruction_losses = [], [], [], []
    for i, (snp_data, suppop_labels, pop_labels) in enumerate(dloader):
        if model.training:
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            assert optimizer is None
        snp_data = snp_data.float().to(device)
        #Make predictions
        loc, scale, embeddings_pred, qzx_fitted, reconstructions = model(snp_data)
        #Calculate various terms in the loss
        total, kl, hyphc, recon = compute_total_loss(model, device, embeddings_pred, reconstructions, 
                                                     snp_data, suppop_labels, pop_labels, kl_weight, 
                                                     hc_weight, recon_weight, sim_func, qzx_fitted)
        total_losses.append(total.item())
        kl_losses.append(kl.item())
        hyphc_losses.append(hyphc.item())
        reconstruction_losses.append(recon.item())
        #If we are training, then update the weights
        if model.training:
            total.backward()
            optimizer.step()
    return np.nanmean(total_losses), np.nanmean(kl_losses), np.nanmean(hyphc_losses), np.nanmean(reconstruction_losses)

def train_model(model, train_loader, valid_loader, num_epochs, learning_rate, sim_func, kl_weight, hc_weight, recon_weight,
                txt_writer, output_dir, early_stopping, patience, early_stop_min_delta, optimizer=None):
    '''
    Code to train VAE model
    Arguments:
        `model: VAE model to be trained
        `train_loader: train dataloader
        `valid_loader: validation dataloader
        `num_epochs: max num epochs to train
        `learning_rate: learning rate for training
        `sim_func: similarity function used to calculate pairwise similarities for HC loss
        `kl_weight (int/float): weight for KL divergence loss
        `hc_weight (int/float): weight for HC loss
        `recon_weight (int/float): weight for reconstruction loss
        `txt_writer: opened csv file to write training logs
        `output_dir: directory to store model and logginig information
        `early_stopping (bool): whether to use early stopping
        `patience (int): num of epochs to store for early stopping
        `early_stop_min_delta (float): min delta for early stopping
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device == torch.device("cuda"):
        print("Training occurring on GPU")
        model = model.to(device)
    if optimizer is None:
        optimizer = RAdam(model.parameters(), lr=learning_rate)
    if early_stopping:
        valid_loss_history = []
    best_valid_epoch_loss, best_model = float("inf"), None
    for epoch in range(num_epochs):
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
        model.train()
        train_epoch_loss, train_kl_loss, train_hyphc_loss, train_recon_loss =\
        run_epoch(model, train_loader, device, sim_func, kl_weight, hc_weight, recon_weight, optimizer)
        print_and_log("Train: epoch %d: total loss = %6.10f, KL loss = %6.10f HC loss = %6.10f, reconstruction loss = %6.10f" %\
                      (epoch + 1, train_epoch_loss, train_kl_loss, train_hyphc_loss, train_recon_loss), txt_writer)
        with torch.no_grad():
            model.eval()
            valid_epoch_loss, valid_kl_loss, valid_hyphc_loss, valid_recon_loss =\
            run_epoch(model, valid_loader, device, sim_func, kl_weight, hc_weight, recon_weight, optimizer=None)
        print_and_log("Valid: epoch %d: total loss = %6.10f, KL loss = %6.10f, HC loss = %6.10f, reconstruction loss = %6.10f" %\
                      (epoch + 1, valid_epoch_loss, valid_kl_loss, valid_hyphc_loss, valid_recon_loss), txt_writer)
        if valid_epoch_loss < best_valid_epoch_loss:
            best_valid_epoch_loss = valid_epoch_loss
            best_model = model
            save_model(model, optimizer, valid_epoch_loss, epoch + 1, output_dir+"model.pt")        
        if early_stopping:
            if len(valid_loss_history) < patience + 1:
                # Not enough history yet; tack on the loss
                valid_loss_history = [valid_epoch_loss] + valid_loss_history
            else:
                # Tack on the new validation loss, kicking off the old one
                valid_loss_history = \
                    [valid_epoch_loss] + valid_loss_history[:-1]
            if len(valid_loss_history) == patience + 1:
                # There is sufficient history to check for improvement
                best_delta = np.max(np.diff(valid_loss_history))
                if best_delta < early_stop_min_delta:
                    break  # Not improving enough
        txt_writer.flush()

def main():
    args = read_config("vae_config.json")
    if not os.path.exists(args["output_dir"]):
        os.mkdir(args["output_dir"])
    os.system("cp vae_config.json %svae_config.json" %(args["output_dir"]))
    print("JSON Copied")
    
    #Load train and validation indices
    train_indices = np.load(args["index_loc"] + "train_indices.npy")
    valid_indices = np.load(args["index_loc"] + "valid_indices.npy")
    
    #Define dataset
    if args["dataset"] == "ancestry":
        dataset_train = HyperLoader(args["data_dir"], train_indices, args["restrict_labels"], args["chromosome"])
        print(len(dataset_train), dataset_train.pop_labels[2])
        dataset_valid = HyperLoader(args["data_dir"], valid_indices, args["restrict_labels"], args["chromosome"])
        print(len(dataset_valid), dataset_valid.pop_labels[2])
        
    #Create train and valid dataloaders
    train_loader = DataLoader(dataset_train, batch_size=args["batch_size"])
    valid_loader = DataLoader(dataset_valid, batch_size=args["batch_size"])
    
    #Define manifolds and models
    manifold = manifold_dict[args["manifold"]](args["embedding_size"])
    enc_type, dec_type  = enc_dec_dict[args["enc_type"]], enc_dec_dict[args["dec_type"]]
    encoder = enc_type(manifold, dataset_train[0][0].shape[-1], args["num_encoder_int_layers"], 
                       args["encoder_int_layer_sizes"], args["encoder_dropout_vals"], args["embedding_size"])
    decoder = dec_type(manifold, dataset_train[0][0].shape[-1], args["num_decoder_int_layers"], 
                       args["decoder_int_layer_sizes"], args["decoder_dropout_vals"], args["embedding_size"])
    
    model = vae_model(encoder, decoder, manifold, distribution_dict[args["posterior_dist"]],
                      distribution_dict[args["prior_dist"]], args["prior_mean"], args["prior_std"], args["temperature"], args["init_size"], args["min_scale"], args["max_scale"])
    print(model)
    #Define logger
    txt_writer = open(args["output_dir"] + "csv_log.csv", "w")
    #Train
    train_model(model, train_loader, valid_loader, args["num_epochs"], args["learning_rate"], 
                sim_func_dict[args["sim_func"]], args["kl_weight"], args["hc_weight"], 
                args["recon_weight"], txt_writer, args["output_dir"], args["early_stopping"], 
                args["patience"], args["early_stopping_min_delta"])
    
if __name__ == "__main__":
    main()

    
                 
    
    
    
    
    

