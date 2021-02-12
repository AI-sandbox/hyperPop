import sys
sys.path.append("../../")
sys.path.append("../../../../libraries/")
import numpy as np
import torch
import os
import json
from torch.utils.data import SubsetRandomSampler, DataLoader
from utils.sim_funcs import sim_func_dict
from utils.model_utils import *
from features.hyperLAIdataset import HyperLoader

data_dir = "/scratch/users/patelas/hyperLAI/snp_data/whole_genome/"
# data_dir = "/scratch/users/patelas/hyperLAI/snp_data/whole_genome/variance_filtered_500000/"

dataset = HyperLoader(data_dir, [0,1,2,3,4,5,6], "all")

train_indices, valid_indices, test_indices = train_valid_test(len(dataset), 0.8, 0.1)

print(test_indices[0]) #Verify that the same indices are always being used

np.random.seed(0)
train_shortened = np.random.choice(train_indices, size=2000)
print("train indices shortened")

variance_filter(dataset, train_shortened, int(sys.argv[1]))

print("Variance Filtered")

np.save("/scratch/users/patelas/hyperLAI/snp_data/whole_genome/variance_filtered_%s/all_snps.npy"%(sys.argv[1]), 
        dataset.snps)


