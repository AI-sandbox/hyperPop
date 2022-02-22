**Hyperbolic Geometry-Based Deep Learning
Methods to Produce Population Trees from
Genotype Data**
==============================


This repo allows for evaluation and training of a variety of models to produce continuous tree representations from genotype data. 

Required packages can be installed by running `pip install -r requirements.txt`

# Evaluation Using Pretrained Models
In our paper, we report results on fully connected (MLP) and variational autoencoder (VAE) models. We provide pretrained versions of these models for users to run inference on their own datasets. 

First, download the [pretrained models](https://drive.google.com/file/d/1se3PHBgG44M_kpzG3WXiPOAzHWtIUsiA/view?usp=sharing) into a directory of your choice, and expand the file. 

The script to run these models is found in `bin/run_hyp_model.py`. The options for this script are as follows:
- `--input_vcf`: The input VCF file consisting of SNP data for the relevant samples. These should contain the 500,000 SNPs in order as detailed in `snps_for_trained_model.tsv` (hg19 coordinates).  
- `--labels`: Text file containing the labels assigned to each data point. Should have one value per line. Labels can be numerical or categorical. 
- `--model_dir`: Directory the trained models are stored in. 
- `--model_type`: Which model to use. Choices are `HypVAE` (hyperbolic VAE model) or `HypMLP` (hyperbolic MLP model).
- `--output_dir`: Directory to store results in.  


The script produces the following output files, which are located in `output_dir`. Note that each haploid genotype is treated separately, so there will be twice as many output data points as input data points. 
- `predicted_embeddings.npy`: numpy array file containing the embeddings for each data point as predicted by the model.
- `embedding_plot.png`: plot of the predicted embeddings. 
- `tree_edges.txt`: file containing the edges of the decoded tree. Each line contains two numbers, representing the indices of the start and end nodes of each (directed) edge. Indices in the range `[0, 2*number of input data points)` represent the genotypes in the data (ie. the last level of the tree), and higher indices represent nodes at higher levels of the tree. 

To successfully run the script, two other repositories should also be installed, and they can be found [here](https://github.com/HazyResearch/HypHC) and [here](https://github.com/emilemathieu/pvae). The script assumes these repos are installed in a folder named `libraries/` at the same location where this repo is installed. If you would like to change this, then you will have to edit the import statements in the script.   

# Training Models
This repo also has functionality to train both the VAE and MLP models. Although this code is specialized for the specific use cases and datasets presented in our paper, it can be adapted fairly easily. 

The first step is to create a folder that contains the following files. For the file size descriptions below, assume `num_individuals` is the number of diploid individuals in the dataset, and `num_genotypes`, or `2 * num_individuals`, is the number of haploid genotypes. 
- `all_snps.npy`: numpy array file containing the SNP data to train and test on. It should be an array of shape `num_genotypes x num_snps`, where each value is a 1 or 0 (as found in a VCF file)
- `labels_pop.npy`: numpy array containing integer population labels for each individual. It should be an array of length `num_individuals`. In our data, these labels correspond to the specific ethnicities of each individual. 
- `labels_suppop.npy`: numpy array containing integer superpopulation labels for each individual. It should be an array of length `num_individuals`. In our data, these labels correspond to the continent each individual is from. 
- `pop_index.npy`: numpy array containing string population labels which correspond to the integer labels found in `labels_pop.npy`. It should be an array of length   `num_individuals`.
- `suppop_code_index.npy`: numpy array containing string superpopulation labels which correspond to the integer labels found in `labels_suppop.npy`. It should be an array of length `num_individuals`.
- `coords.npy` and `suppop_index.npy` are legacy requirements and not used. You can just use dummy numpy arrays for these. 

Next, you will need to define train, validation, and test splits for the data. To do this, create a folder and populate it with `train_indices.npy`, `valid_indices.npy`, and `test_indices.npy`. All indices from 0 to `num_genotypes` should be included in one of these three files. 

Finally, the appropriate config file will need to be filled out. For the VAE model, this is `hyperLAI/models/vae_config.json`, and for the MLP model, this is `hyperLAI/models/fc_config.json`. Examples and explanations of the various parameters can be found [here](https://drive.google.com/file/d/1mh9AwTuG2m7Raqa_M0cO8BOhsE3KooLN/view?usp=sharing) and [here](https://drive.google.com/file/d/171xQdnj45nnNkGyc0gL49-Y-i19v1FFY/view?usp=sharing).  

After this, simply run `python train_vae_model.py` or `python train_fc_model.py` from the same folder. 

One important modeling choice concerns the similarity function used. Currently, we use a simple function which calculates the fraction of SNPs shared between two individuals. However, this can be customized - the training script supports any similarity function on the SNP data, and it can be adapted to support similarity functions on the labels as well. See `hyperLAI/utils/sim_funcs.py` for more information. 

