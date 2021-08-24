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

#This code takes in a genotype dataset and filters to only include the SNPs with the k highest variances across the population

#Define the directory to draw data from
data_dir = "/scratch/users/patelas/hyperLAI/snp_data/whole_genome/"
train_inds = np.load("/scratch/users/patelas/hyperLAI/ancestry_training_splits/80_10_10/train_indices.npy")
valid_inds = np.load("/scratch/users/patelas/hyperLAI/ancestry_training_splits/80_10_10/valid_indices.npy")
test_inds = np.load("/scratch/users/patelas/hyperLAI/ancestry_training_splits/80_10_10/test_indices.npy")
all_inds = np.sort(np.concatenate([train_inds, valid_inds, test_inds]))
print(all_inds[0], all_inds[-1])
pop_labels = [3]
output_dir = "/scratch/users/patelas/hyperLAI/snp_data/whole_genome/variance_filtered_500000_subpops/south_asian/"



#Create the dataset
dataset = HyperLoader(data_dir, all_inds, [0,1,2,3,4,5,6], "all")

#Get indices to use
pop_indices = np.argwhere(np.isin(dataset.suppop_labels, pop_labels)).T[0]
indices = np.intersect1d(pop_indices, train_inds)
print(len(indices))
print(indices[0], indices[-1])


#Filter by variance
variance_filter(dataset, indices, int(sys.argv[1]))

print("Variance Filtered")

#Save to file
np.save(output_dir + "all_snps.npy", 
        dataset.snps)

#Note: This script only creates a copy of all_snps.npy in the desired folder. All other metadata files will have to be copied manually. 
