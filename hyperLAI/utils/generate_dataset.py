import allel
import time
from collections import Counter, OrderedDict
import gzip
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.cm as cm
from operator import itemgetter
import os
import sys
import re
import torch
import numpy as np
from collections import defaultdict
import pandas as pd
import random
sys.path.append("../")
from utils.vcf_utils import read_vcf, vcf_to_npy_simple



def load_dataset(vcf_path, tsv_path, output_path, chromosome=22, verbose=True, filter_admixed=True, filter_missing_coord=True):

    if verbose:
        print('start reading ...')

    #Reading input vcf file
    mat_vcf_2d, vcf_pos, vcf_data_names = vcf_to_npy_simple(vcf_path, chromosome, reshape=False)

    if verbose:
        print('done loading vcf...', 'Shape of vcf file is : ', mat_vcf_2d.shape, 'Shape of vcf names file is', vcf_data_names.shape)

    #Reading TSV metadata (ground-truth labels)
    df = pd.read_csv(tsv_path, sep='\t')
    
    if verbose:
        print('TSV loaded', df.columns, 'DF shape is ', df.shape)

    SUPPOP_CODE = list(df['Superpopulation code'].unique())  #Continental population code (i.e. AFR, EAS, ...)
    POP_CODE = list(df['Population code'].unique()) #Country/Local population code (i.e. YRI, ESP, ...)
    POP_NAME = list(df['Population'].unique()) #Country/Local population name (i.e. Basque, Han Chinese, ...)

    if verbose:
        print('Super population code are , ', SUPPOP_CODE, len(SUPPOP_CODE))
        print('Population code are , ', POP_CODE, len(POP_CODE))
        print('Population names are , ', POP_NAME, len(POP_NAME))


    N = len(vcf_data_names)


    # Parsing TSV file (TODO: requires cleaning and optimization)
    labels = np.zeros(N) - 1
    labels_subcontinent = np.zeros(N) - 1
    coords = np.zeros((N, 2)) - 1

    
    valid_idx = []
    j = 0
    for i, name in enumerate(vcf_data_names): 
        if name in df['Sample'].unique():

            is_single = list(df.loc[df.Sample==name, 'Single_Ancestry'])[0]


            lat = list(df.loc[df.Sample==name, 'Latitude'])[0]
            lon = list(df.loc[df.Sample==name, 'Longitude'])[0]

            
            valid = True
            
            if filter_admixed and is_single == 0:
                valid = False
                
            if filter_missing_coord and np.isnan(lat):
                valid = False
                
            if valid:
                valid_idx.append(j)
               

            coords[j, 0] = float(lat)
            coords[j, 1] = float(lon)

            sup_code = list(df.loc[df.Sample==name, 'Superpopulation code'])[0]
            label = SUPPOP_CODE.index(str(sup_code))

            labels[j] = label

            pop_code = list(df.loc[df.Sample==name, 'Population'])[0]
            l = POP_NAME.index(pop_code)
            labels_subcontinent[j] = l

            j+=1


    # Selecting valid entries (single ancestry and no missing coordinates)
    labels = labels[valid_idx]
    coords = coords[valid_idx,:]
    labels_subcontinent = labels_subcontinent[valid_idx]
    mat = mat_vcf_2d[:,valid_idx,:]


    return mat, labels, labels_subcontinent, coords, POP_NAME, POP_CODE, SUPPOP_CODE

def sort_ancestries_by_size(labels_subcontinent, POP_NAME, verbose=True):
    count, _ = np.histogram(labels_subcontinent, bins=len(POP_NAME))
    unique_labels = np.unique(labels_subcontinent).astype(int)
    
    ranked = np.argsort(count)
    largest_indices = ranked[::-1]
    
    POP_NAME = np.array(POP_NAME)
    largest_indices = np.array(largest_indices)
    
    POP_NAME_SORTED = POP_NAME[largest_indices]
    
    assert len(POP_NAME) == len(POP_NAME_SORTED)
    
    labels_pops_names = POP_NAME[labels_subcontinent.astype(int)]
    
    labels_subcontinent_sorted = []
    for k in range(len(labels_subcontinent)):
        labels_subcontinent_sorted.append(list(POP_NAME_SORTED).index(labels_pops_names[k]))
  

    assert len(labels_subcontinent_sorted) == len(labels_subcontinent)
    
    
    if verbose:
        plt.hist(labels_subcontinent, bins=len(POP_NAME))
        plt.show()
        
        plt.hist(labels_subcontinent_sorted, bins=len(POP_NAME))
        plt.show()
    
    return np.array(labels_subcontinent_sorted), list(POP_NAME_SORTED)

def filter_small_ancestries(mat, labels, labels_subcontinent, coords, POP_NAME, verbose=True, threshold=1):
    
    # FILTER ANCESTRY PER NUMBER OF INDIVIDUALS -------------------------------
    # Keeping ancestries with ''threshold'' or more individuals
    
    if verbose:
        plt.hist(labels_subcontinent, bins=len(POP_NAME))
        plt.show()

    count, _ = np.histogram(labels_subcontinent, bins=len(POP_NAME))
    unique_labels = np.unique(labels_subcontinent).astype(int)

    large_pops = []
    for elem in unique_labels:
        if count[elem] >= threshold:
            if verbose:
                print(POP_NAME[elem], count[elem])
            large_pops.append(elem)

    if verbose:
        print(len(large_pops), large_pops)

    large_pops_idx = []
    for idx, elem in enumerate(labels_subcontinent):
        if elem in large_pops:
            large_pops_idx.append(idx)

    labels = labels[large_pops_idx]
    coords = coords[large_pops_idx,:]
    labels_subcontinent = labels_subcontinent[large_pops_idx]
    mat = mat[:,large_pops_idx,:]
    
    
    # Re-label population names and indexs (so there is no index with 0 individuals)
    unique_subopops = np.unique(labels_subcontinent)
    POP_UNIQUE_NAME = []

    for elem in unique_subopops:
        POP_UNIQUE_NAME.append(POP_NAME[int(elem)])

    if verbose:
        print('Large populations are', POP_UNIQUE_NAME, len(POP_UNIQUE_NAME))

    labels_subcontinent_new = []
    for elem in labels_subcontinent:
        labels_subcontinent_new.append(list(unique_subopops).index(elem))
    labels_subcontinent_new = np.array(labels_subcontinent_new)

    count, _ = np.histogram(labels_subcontinent_new, bins=len(POP_UNIQUE_NAME))
    unique_labels = np.unique(labels_subcontinent_new).astype(int)

    if verbose:
        for elem in unique_labels:
            print(POP_UNIQUE_NAME[elem], count[elem])

    if verbose:
        print('done')
    
    return mat, labels, labels_subcontinent_new, coords, POP_UNIQUE_NAME






def generate_train_val_test_splits(mat, labels, labels_subcontinent, coords, POP_NAME, haploid=True, shuffle=True):

    count, _ = np.histogram(labels_subcontinent, bins=len(POP_NAME))

    unique_labels = np.unique(labels_subcontinent).astype(int)

    ## Train / val / test split - 80 / 10 / 10
    train_idx = []
    val_idx = []
    test_idx = []

    for j, elem in enumerate(unique_labels):
        assert j == elem
        
        idx_list_pop_j = np.where(labels_subcontinent == elem)[0]
        
        if shuffle:
            np.random.shuffle(idx_list_pop_j)

        
        assert len(idx_list_pop_j) == count[j]

        k_train, k_val, k_test = 0, 0, 0
        for idx in list(idx_list_pop_j):

            if k_test <= count[j]*0.1:
                test_idx.append(idx)
                k_test += 1
            elif k_val <= count[j]*0.1:
                val_idx.append(idx)
                k_val += 1
            else:
                train_idx.append(idx)
                k_train += 1

    #print(len(train_idx), len(val_idx), len(test_idx))



    labels_suppop_train = labels[train_idx]
    coords_train = coords[train_idx,:]
    labels_train = labels_subcontinent[train_idx]
    dset_train = mat[:,train_idx,:]
    #print(labels_suppop_train.shape, coords_train.shape, labels_train.shape, dset_train.shape)


    labels_suppop_val = labels[val_idx]
    coords_val = coords[val_idx,:]
    labels_val = labels_subcontinent[val_idx]
    dset_val = mat[:,val_idx,:]
    #print(labels_suppop_val.shape, coords_val.shape, labels_val.shape, dset_val.shape)
    
    
    labels_suppop_test = labels[test_idx]
    coords_test = coords[test_idx,:]
    labels_test = labels_subcontinent[test_idx]
    dset_test = mat[:,test_idx,:]
    #print(labels_suppop_test.shape, coords_test.shape, labels_test.shape, dset_test.shape)

    #print(dset_train.dtype)
    
    if haploid:
        dset_train = np.sum(dset_train, axis=2).T #.astype(np.unitc)
        dset_val = np.sum(dset_val, axis=2).T
        dset_test = np.sum(dset_test, axis=2).T
        
    #print(dset_train.dtype, dset_train.shape, dset_train.min(), dset_train.max())

    
    train = (dset_train, labels_train, coords_train, labels_suppop_train)
    val = (dset_val, labels_val, coords_val, labels_suppop_val)
    test = (dset_test, labels_test, coords_test, labels_suppop_test)

#     if False:
#         dset_train = torch.tensor(mat_train).to(device).float()
#         dset_train = torch.mean(dset_train, dim=2)

#         dset_val = torch.tensor(mat_val).to(device).float()
#         dset_val = torch.mean(dset_val, dim=2)
#         print(dset_val.dtype, dset_val.shape, dset_val.min(), dset_val.max())

#         dset_train = dset_train.T
#         dset_val = dset_val.T

        # TODO: save mean per category?
    return train, val, test





def generate_and_save_dataset(founders_path, anc_panel, output_path, chromosome=22, threshold_individuals_per_ancestry=1):
    print('new!')
    print('Loading and processing vcf file...')
    output = load_dataset(founders_path, anc_panel, output_path, chromosome=chromosome, verbose=False, filter_admixed=True, filter_missing_coord=True)
    mat, labels, labels_subcontinent, coords, POP_NAME, POP_CODE, SUPPOP_CODE = output
    
    print('Sort populations by size...')
    labels_subcontinent, POP_NAME = sort_ancestries_by_size(labels_subcontinent, POP_NAME)

    print('Filtering small ancestry groups...')
    mat, labels, labels_subcontinent, coords, POP_NAME = filter_small_ancestries(mat, labels, labels_subcontinent, coords, POP_NAME, threshold=threshold_individuals_per_ancestry, verbose=True)

    plt.hist(labels_subcontinent, bins=len(POP_NAME))
    plt.show()
    
    print('Generating train/val/test splits...')
    train, val, test = generate_train_val_test_splits(mat, labels, labels_subcontinent, coords, POP_NAME)
    
    print('Saving processed files...')
    dataset = list(train) + list(val) + list(test) + list([np.array(POP_NAME)]) + list([np.array(SUPPOP_CODE)])

    dataset_fname = os.path.join(output_path, 'dataset.npz')
    np.savez_compressed(dataset_fname, *dataset)
    full_dataset = np.load(dataset_fname)
    
    print('Done! Dataset saved as: {}'.format(dataset_fname))
    
    return dataset_fname



def load_npz_dataset(dataset_fname):
    full_dataset = np.load(dataset_fname)
    array_list = []
    for k in full_dataset:
        print(k)
        array_list.append(full_dataset[k])

    dset_train, labels_train, coords_train, labels_suppop_train, dset_val, labels_val, coords_val, labels_suppop_val, dset_test, labels_test, coords_test, labels_suppop_test, POP_NAME, SUPPOP_CODE = array_list
    
    train = (dset_train, labels_train, coords_train, labels_suppop_train)
    val = (dset_val, labels_val, coords_val, labels_suppop_val)
    test = (dset_test, labels_test, coords_test, labels_suppop_test)
    
    return train, val, test, POP_NAME, SUPPOP_CODE

