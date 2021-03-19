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

#Create the dataset
dataset = HyperLoader(data_dir, [0,1,2,3,4,5,6], "all")

#Split indices into train, valid, and test (since we only want to do calculations on the train data)
#This function produces reproducible results
train_indices, valid_indices, test_indices = train_valid_test(len(dataset), 0.8, 0.1)

print(test_indices[0]) #Verify that the same indices are always being used

#In practice, due to time/space constraints, I used a sample of 2000 training samples for variance calcs
np.random.seed(0)
train_shortened = np.random.choice(train_indices, size=2000)
print("train indices shortened")

#Filter by variance
variance_filter(dataset, train_shortened, int(sys.argv[1]))

print("Variance Filtered")

#Save to file
np.save("/scratch/users/patelas/hyperLAI/snp_data/whole_genome/variance_filtered_%s/all_snps.npy"%(sys.argv[1]), 
        dataset.snps)

#Note: This script only creates a copy of all_snps.npy in the desired folder. All other metadata files will have to be copied manually. 
