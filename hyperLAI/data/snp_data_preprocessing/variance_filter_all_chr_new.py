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

#Define the direectory to draw data from
data_dir = "/scratch/users/patelas/hyperLAI/snp_data/whole_genome/"
train_inds = np.load("/scratch/users/patelas/hyperLAI/ancestry_training_splits/80_10_10/train_indices.npy")
valid_inds = np.load("/scratch/users/patelas/hyperLAI/ancestry_training_splits/80_10_10/valid_indices.npy")
test_inds = np.load("/scratch/users/patelas/hyperLAI/ancestry_training_splits/80_10_10/test_indices.npy")
all_inds = np.sort(np.concatenate([train_inds, valid_inds, test_inds]))
print(all_inds[0], all_inds[-1])

#Create the dataset
dataset = HyperLoader(data_dir, all_inds, [0,1,2,3,4,5,6], "all")
print(len(dataset))

#In practice, due to time/space constraints, I used a sample of 2000 training samples for variance calcs
np.random.seed(0)
train_shortened = np.random.choice(train_inds, size=2000, replace=False)
print("train indices shortened")

#Filter by variance
variance_filter(dataset, train_shortened, int(sys.argv[1]))

print("Variance Filtered")

#Save to file
np.save("/scratch/users/patelas/hyperLAI/snp_data/whole_genome/variance_filtered_%s_updated/all_snps.npy"%(sys.argv[1]), 
        dataset.snps)

#Note: This script only creates a copy of all_snps.npy in the desired folder. All other metadata files will have to be copied manually. 
