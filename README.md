**Hyperbolic Geometry-Based Deep Learning
Methods to Produce Population Trees from
Genotype Data**
==============================


This repo allows for training and evaluation of a variety of models to produce continuous tree representations from genotype data. 

Required packages can be installed by running `pip install -r requirements.txt`

# Evaluation Using Pretrained Models
In our paper, we report results on fully connected (MLP) and variational autoencoder (VAE) models. We provide pretrained versions of these models for users to run inference on their own datasets. The script to run these models is found in `bin/run_hyp_model.py`. The options for this script are as follows:
- `--input_vcf`: The input VCF file consisting of SNP data for the relevant samples. These should contain the 500,000 SNPs in order as detailed in THIS FILE (hg19 coordinates).  
- `--labels`: Text file containing the labels assigned to each data point. Should have one value per line. Labels can be numerical or categorical. 
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
- `labels_pop.npy`: numpy array containing population labels for each individual. It should be an array of length `num_individuals`. In our data, these labels correspond to the specific ethnicities of each individual. 
- `labels_suppop.npy`: numpy array containing superpopulation labels for each individual. It should be an array of length `num_individuals`. In our data, these labels correspond to the continent each individual is from. 
- 


- HypHC Model ([Chami et al. 2020](https://arxiv.org/pdf/2010.00402.pdf)) - use `notebooks/train_simple_model.ipynb` with the appropriate parameter specifications
- MLP (fully connected model) - first, navigate to `hyperLAI/models`. Then, configure the parameters and other specifications in fc_config.json` Finally, run `python train_fc_model.py`
- VAE (variational autoencoder) - first, navigate to `hyperLAI/models`. Then, configure the parameters and other specifications in vae_config.json` Finally, run `python train_vae_model.py`




To evaluate the MLP model and visualize its output, use `notebooks/eval_fc_model.ipynb`, and to do the same for the VAE model, use `notebooks/eval_vae_model.ipynb`
