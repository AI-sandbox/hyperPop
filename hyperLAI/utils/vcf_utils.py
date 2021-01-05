import allel
import time
from collections import Counter, OrderedDict
import gzip

def read_vcf(vcf_file, verb=True):
    """
    Reads vcf files into a dictionary
    fields="*" extracts more information, take out if ruled unecessary
    """
    if vcf_file[-3:]==".gz":
        with gzip.open(vcf_file, 'rb') as vcf:
            data = allel.read_vcf(vcf) #, fields="*")
    else: 
        data = allel.read_vcf(vcf_file) #, fields="*")
    if verb:    
        chmlen, n, _ = data["calldata/GT"].shape
        print("File read:", chmlen, "SNPs for", n, "individuals")
    return data

def vcf_to_npy_simple(vcf_fname, chm, reshape=True, boolean=True):
    """
    Converts vcf file to numpy matrix. If SNP position format is specified, then
    accompany that format by filling in values of missing positions and ignoring
    additional positions.
    """
    
    # unzip and read vcf
    vcf_data = read_vcf(vcf_fname)
    chm_idx = vcf_data['variants/CHROM']==str(chm)
    # matching SNP positions with standard format (finding intersection)
    vcf_pos = vcf_data['variants/POS'][chm_idx]
    #fmt_idx, vcf_idx = snp_intersection(snp_pos_fmt, vcf_pos, verbose=verbose)
    
    # reshape binary represntation into 2D np array 
    chm_data = vcf_data["calldata/GT"][:,:,:]
    chm_len, nout, _ = chm_data.shape
    if reshape:
        chm_data = chm_data.reshape(chm_len,nout*2).T
    if boolean:
        chm_data = chm_data.astype(dtype=bool) #np.array(chm_data, dtype=bool)
    
    mat_vcf_2d = chm_data
    return mat_vcf_2d, vcf_pos, vcf_data['samples']