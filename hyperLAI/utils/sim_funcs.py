import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score

def mnist_label_evenodd(x1,x2):
    '''
    Similarity function used for MNIST experiments
    Adds one point if labels are the same and another point if labels are both even/odd
    '''
    same_label = int(x1 == x2)
    same_mod = int((x1 % 2) == (x2 % 2))
    return (same_label + same_mod) / 2

def ancestry_label_sim(labels_1, labels_2):
    suppop_label = int(labels_1[0] == labels_2[0])
    pop_label = int(labels_1[1] == labels_2[1])
    return (suppop_label + pop_label) / 2

def ancestry_label_sim_subpop(labels_1, labels_2):
    pop_label = int(labels_1[1] == labels_2[1])
    return pop_label


#hammming: fraction of SNPs that are shared
sim_func_dict = {
    "hamming": lambda x,y: torch.true_divide(torch.sum(x==y), len(x)),
    "minf": adjusted_mutual_info_score, 
    "mnist_even_odd": mnist_label_evenodd,
    "ancestry_label": ancestry_label_sim,
    "ancestry_label_subpop": ancestry_label_sim_subpop,
}