import torch
import numpy as np
import itertools
import json

def make_pairwise_similarities(snp_data, sim_func):
    sim_matrix = torch.zeros((snp_data.shape[0], snp_data.shape[0]))    
    for ind in range(snp_data.shape[0]):
        for ind2 in range(snp_data.shape[0]):
            sim_matrix[ind][ind2] = sim_func(snp_data[ind], snp_data[ind2])
    sim_matrix /= torch.max(sim_matrix)
    return sim_matrix

def generate_triple_ids(num_inds):
    return list(itertools.combinations(list(range(num_inds)), 3))
                        
def trips_and_sims(snp_data, sim_func):
    triple_ids = generate_triple_ids(snp_data.shape[0])
    sim_matrix = make_pairwise_similarities(snp_data, sim_func)
    sim_vals = [[sim_matrix[tr[0], tr[1]],sim_matrix[tr[0], tr[2]],sim_matrix[tr[1], tr[2]]] for tr in triple_ids]
    return torch.tensor(triple_ids), torch.tensor(sim_vals)

def save_model(model, optimizer, validation_loss, epoch, save_path):
    """
    Saves the given model at the given path. This saves the state of the model
    (i.e. trained layers and parameters), and the arguments used to create the
    model (i.e. a dictionary of the original arguments). 
    This is used to save models at the END of epochs. 
    """
    save_dict = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss
    }
    torch.save(save_dict, save_path)

def print_and_log(string, file):
    print(string)
    file.write(string.replace(": ", ",").replace(" = ", ",") + "\n")
    
def read_config(config_path):
    json_file = open(config_path)
    config_vars = json.load(json_file)
    json_file.close()
    return config_vars
    
def train_valid_test(data_len, train_perc, valid_perc):
    train_size = int(train_perc * data_len)
    valid_size = int(valid_perc * data_len)
    test_size = data_len - train_size - valid_size
    np.random.seed(0)
    indices = np.array(list(range(data_len)))
    np.random.shuffle(indices)
    train_indices = indices[0 : train_size]
    valid_indices = indices[train_size : train_size + valid_size]
    test_indices = indices[train_size + valid_size : ]
    return train_indices, valid_indices, test_indices

def variance_filter(dataset, train_indices, snps_to_keep):
    train_vars = np.var(dataset.snps[train_indices, :], axis=0)
    print("Variances Calculated")
    snps_preserved = np.argsort(train_vars)[::-1][0:snps_to_keep]
    print("Subset calculated")
    dataset.snps = dataset.snps[:,snps_preserved]
    return
                          
                            
                        
