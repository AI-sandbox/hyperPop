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
    return same_label + same_mod

#hammming: fraction of SNPs that are shared
sim_func_dict = {
    "hamming": lambda x,y: torch.true_divide(torch.sum(x==y), len(x)),
    "minf": adjusted_mutual_info_score, 
    "mnist_even_odd": mnist_label_evenodd
}