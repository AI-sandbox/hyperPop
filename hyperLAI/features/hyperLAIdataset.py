import numpy as np
import torch
from torch.utils import data
import sys
from utils.generate_dataset import *
from HypHC.datasets.triples import samples_triples

class HyperLoader(data.Dataset):
    def __init__(self, data_dir, restrict_labels=[0,1,2,3,4,5,6], chromosome="all"):
        '''
        Takes in all the relevant arguments to produce the dataset.
        Arguments:
            `data_dir`: directory in which data (either text files or numpy arrays) are located
            `similarity_func`: function to calculate pairwise similarities
            `restrict_labels`: list of super-populations to include in analysis. Indices correspond to 'EUR', 'EAS', 'AMR', 'SAS', 'AFR', 'OCE', 'WAS'
        '''

        self.data_dir = data_dir 
        self.restrict_labels = restrict_labels
        self.chromosome = chromosome
        self.snps, self.pop_labels, self.suppop_labels, self.pop_label_index, self.suppop_label_index = self.load_data()
    def load_data(self):
        '''
        Loads SNP and label data from the necessary file locations 
        '''
        #If we want all chromosomes, then we have the arrays already pre-created
        if self.chromosome =="all":
            file_order = ["all_snps.npy", "labels_suppop.npy", "labels_pop.npy", 
                          "coords.npy", "pop_index.npy", "pop_code_index.npy", "suppop_code_index.npy"]
            test_data = tuple([np.load(self.data_dir + x) for x in file_order])
            ind_data = test_data[0]
        else:
            #The data for individual chromosomes is in a slightly different format
            test_data = load_dataset(self.data_dir + "ref_final_beagle_phased_1kg_hgdp_sgdp_chr%s_hg19.vcf.gz"%(self.chromosome), 
                                     self.data_dir + "reference_panel_metadata.tsv", "./", chromosome=self.chromosome, 
                                     verbose=True, filter_admixed=True, filter_missing_coord=True)
            ind_data = test_data[0].reshape([test_data[0].shape[0], test_data[0].shape[1] * test_data[0].shape[2]]).T            
        #We've unfolded each set of 23 chromosomes as a "different" individual 
        #So we must do the same for the labels by doubling them
        ind_pop_labels = np.repeat(test_data[2], 2).astype(int)
        ind_suppop_labels = np.repeat(test_data[1], 2).astype(int)
        #Restrict to only the super-populations we've specified
        indices = np.argwhere(np.isin(ind_suppop_labels, self.restrict_labels)).T[0]
        #Return everything
        return ind_data[indices], ind_pop_labels[indices], ind_suppop_labels[indices], test_data[4], test_data[6]
    def __len__(self):
        return len(self.snps)
    def __getitem__(self, index):
        '''
        Returns data and labels for the current index
        '''
        return torch.tensor(self.snps[index]), torch.tensor(self.suppop_labels[index]), torch.tensor(self.pop_labels[index])
