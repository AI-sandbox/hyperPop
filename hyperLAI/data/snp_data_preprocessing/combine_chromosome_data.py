import sys
import numpy as np
sys.path.append("../../")
from utils.generate_dataset import *

tsv_path = "/scratch/users/patelas/hyperLAI/reference_panel_metadata.tsv"                    
output_path = "./"                  
stem_path = "/scratch/users/patelas/hyperLAI/"

data_out_stem = "/scratch/users/patelas/hyperLAI/whole_genome/"
data_list = []
for chrom in range(1,23):
    path = stem_path + "ref_final_beagle_phased_1kg_hgdp_sgdp_chr%s_hg19.vcf.gz"%(str(chrom))
    curr_chr_data = load_dataset(path, tsv_path, output_path, chromosome=chrom, 
                                 verbose=True, filter_admixed=True, filter_missing_coord=True)
    data_list.append(curr_chr_data[0])
    metadata = curr_chr_data[1:]

snp_data_all_chr = np.concatenate(data_list, axis=0)
ind_data = snp_data_all_chr.reshape([snp_data_all_chr.shape[0], 
                                        snp_data_all_chr.shape[1] * snp_data_all_chr.shape[2]]).T

np.save(data_out_stem+"all_snps.npy", ind_data) 
np.save(data_out_stem+"labels_suppop.npy", metadata[0])
np.save(data_out_stem+"labels_pop.npy", metadata[1])
np.save(data_out_stem+"coords.npy", metadata[2])
np.save(data_out_stem+"pop_index.npy", np.array(metadata[3]))
np.save(data_out_stem+"pop_code_index.npy", np.array(metadata[4]))
np.save(data_out_stem+"suppop_code_index.npy", np.array(metadata[5]))
