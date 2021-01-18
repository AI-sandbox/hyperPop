import sys
sys.path.append("../")
sys.path.append("../../../")
import numpy as np
import torch
import os
import json
from torch.utils.data import SubsetRandomSampler, DataLoader
from utils.sim_funcs import sim_func_dict
from utils.model_utils import *
from features.hyperLAIdataset import HyperLoader
from models.fc_model import fc_model
from HypHC.optim.radam import RAdam

def run_epoch(model, dloader, device, sim_func, optimizer=None):
    batch_hyp_losses = []
    for i, (snp_data, suppop_labels, pop_labels) in enumerate(dloader):
        if model.training:
            assert optimizer is not None
            optimizer.zero_grad()
        else:
            assert optimizer is None
        triple_ids, similarities = trips_and_sims(snp_data, sim_func)
        snp_data = snp_data.float().to(device)
        triple_ids = triple_ids.to(device)
        similarities = similarities.float().to(device)
        embeddings_pred = model(snp_data)
        hyp_loss = model.calculate_loss(embeddings_pred, triple_ids, similarities)
        batch_hyp_losses.append(hyp_loss.item())
        if model.training:
            hyp_loss.backward()
            optimizer.step()
    return batch_hyp_losses
        
def train_model(model, train_loader, valid_loader, num_epochs, learning_rate, sim_func,
               txt_writer, output_dir, early_stopping, 
                patience, early_stop_min_delta, optimizer=None):
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
        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        model.train()
        train_batch_losses = run_epoch(model, train_loader, device, sim_func, optimizer)
        train_epoch_loss = np.nanmean(train_batch_losses)
        print_and_log("Train: epoch %d: average loss = %6.10f" % (epoch + 1, train_epoch_loss), txt_writer)
        with torch.no_grad():
            model.eval()
            valid_batch_losses = run_epoch(model, valid_loader, device, sim_func, optimizer=None)
        valid_epoch_loss = np.nanmean(valid_batch_losses)
        print_and_log("Valid: epoch %d: average loss = %6.10f" % (epoch + 1, valid_epoch_loss), txt_writer)
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
    config = read_config("fc_config.json")
    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])
    os.system("cp fc_config.json %sfc_config.json" %(config["output_dir"]))
    print("JSON Copied")
    dataset = HyperLoader(config["data_dir"], config["restrict_labels"], config["chromosome"])
    
    train_indices, valid_indices, test_indices = train_valid_test(len(dataset), config["train_perc"], config["valid_perc"])
    np.save(config["output_dir"] + "train_indices.npy", train_indices)
    np.save(config["output_dir"] + "valid_indices.npy", valid_indices)
    np.save(config["output_dir"] + "test_indices.npy", test_indices)
    
    if config["variance_filter"] is not None:
        variance_filter(dataset, train_indices, config["variance_filter"])
        
    train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=valid_sampler)
    model = fc_model(dataset.snps.shape[1], config["num_int_layers"], config["int_layer_sizes"], config["embedding_size"], 
                     config["dropout_vals"], config["temperature"], config["init_size"], config["min_scale"], config["max_scale"])
    txt_writer = open(config["output_dir"] + "csv_log.csv", "w")
    
    train_model(model, train_loader, valid_loader, config["num_epochs"], config["learning_rate"], sim_func_dict[config["sim_func"]],
                txt_writer, config["output_dir"], config["early_stopping"], config["patience"], config["early_stopping_min_delta"])
    
if __name__ == "__main__":
    main()
    