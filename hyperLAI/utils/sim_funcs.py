import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score

sim_func_dict = {
    "hamming": lambda x,y: torch.true_divide(torch.sum(x==y), len(x)),
    "minf": adjusted_mutual_info_score
}