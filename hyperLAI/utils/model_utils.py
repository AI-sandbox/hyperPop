import torch
import numpy as np
import itertools
import json

def make_pairwise_similarities(snp_data, sim_func):
    '''
    Makes a pairwise similarity matrix
    Arguments:
        `snp_data: data to be used for the similarity comparison (somewhat outdated variable name)
        `sim_func: similarity function which takes in two instances of whatever data type makes up snp_data
    '''
    sim_matrix = torch.zeros((snp_data.shape[0], snp_data.shape[0]))    
    for ind in range(snp_data.shape[0]):
        for ind2 in range(snp_data.shape[0]):
            sim_matrix[ind][ind2] = sim_func(snp_data[ind], snp_data[ind2])
    sim_matrix /= torch.max(sim_matrix) #Might consider removing this line 
    return sim_matrix

def generate_triple_ids(num_inds):
    '''
    Returns all possible triples of num_inds indices
    '''
    return list(itertools.combinations(list(range(num_inds)), 3))
                        
def trips_and_sims(snp_data, sim_func):
    '''
    Returns triples along with all-pairs similarity values for each triple
    To illustrate: similarity values for triple [0,1,2] will be [sim_func(snp_data[0], snp_data[1]), sim_func(snp_data[0], snp_data[2], sim_func(snp_data[1], snp_data[2]))
    '''
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
    '''
    Handy function to print something to a terminal and write it to a log file
    '''
    print(string)
    file.write(string.replace(": ", ",").replace(" = ", ",") + "\n")
    
def read_config(config_path):
    '''
    Reads a config json file and returns it as a dictionary
    '''
    json_file = open(config_path)
    config_vars = json.load(json_file)
    json_file.close()
    return config_vars
    
def train_valid_test(data_len, train_perc, valid_perc):
    '''
    Divides indices in a dataset into train, valid, and test
    Specifying the random seed makes it reproducible
    Arguments:
     `data_len: int - length of dataset
     `train_perc: float - fraction of dataset to be used in training
     `valid_perc: float - fraction to be used in validation
    Returns: np arrays for the train, valid, and test indices
    '''
    #Calculate sizes of all three datasets
    train_size = int(train_perc * data_len)
    valid_size = int(valid_perc * data_len)
    test_size = data_len - train_size - valid_size
    np.random.seed(0) #Could consider making this a parameter
    indices = np.array(list(range(data_len)))
    #Shuffle indices and then divide them
    np.random.shuffle(indices)
    train_indices = indices[0 : train_size]
    valid_indices = indices[train_size : train_size + valid_size]
    test_indices = indices[train_size + valid_size : ]
    return train_indices, valid_indices, test_indices

def variance_filter(dataset, train_indices, snps_to_keep):
    '''
    Given a dataset, uses the train data to only keep the k columns with the highest variance
    Arguments:
        `dataset: torch dataset (although this function was specifically intended for the genotype data)
        `train_indices: indices of the training data
        `snps_to_keep: number of columns to keep
    '''
    train_vars = np.var(dataset.snps[train_indices, :], axis=0)
    print("Variances Calculated")
    snps_preserved = np.argsort(train_vars)[::-1][0:snps_to_keep]
    print("Subset calculated")
    dataset.snps = dataset.snps[:,snps_preserved]
    return
                          
                            
                        
