**Hyperbolic Geometry-Based Deep Learning
Methods to Produce Population Trees from
Genotype Data**
==============================


This repo allows for training and evaluation of a variety of models to produce continuous tree representations from genotype data. 

In our paper, we report results on fully connected (MLP) and variational autoencoder (VAE) models. We provide pretrained versions of these models for users to run inference on their own datasets. The script to run these models is found in `bin/run_hyp_model.py`. The options for this script are as follows:
- `--input_vcf`: The input VCF file consisting of SNP data for the relevant samples. These should contain the 500,000 SNPs in order as detailed in THIS FILE (hg19 coordinates).  
- `--labels`: Text file containing the labels assigned to each data point. Should have one value per line. Labels can be numerical or categorical. 
- `--model_type`: Which model to use. Choices are `HypVAE` (hyperbolic VAE model) or `HypMLP` (hyperbolic MLP model).
- `--output_dir`: Directory to store results in.
The script produces the following output files, which are located in `output_dir`. Note that each haploid genotype is treated separately, so there will be twice as many output data points as input data points. 
- `predicted_embeddings.npy`: numpy array file containing the embeddings for each data point as predicted by the model.
- `tree_edges.txt`: file containing the edges of the decoded tree. Each line contains two numbers, representing the indices of the start and end nodes of each (directed) edge. Indices in the range `[0, 2*number of input data points)` represent the genotypes in the data (ie. the last level of the tree), and higher indices represent nodes at higher levels of the tree. 




- HypHC Model ([Chami et al. 2020](https://arxiv.org/pdf/2010.00402.pdf)) - use `notebooks/train_simple_model.ipynb` with the appropriate parameter specifications
- MLP (fully connected model) - first, navigate to `hyperLAI/models`. Then, configure the parameters and other specifications in fc_config.json` Finally, run `python train_fc_model.py`
- VAE (variational autoencoder) - first, navigate to `hyperLAI/models`. Then, configure the parameters and other specifications in vae_config.json` Finally, run `python train_vae_model.py`




To evaluate the MLP model and visualize its output, use `notebooks/eval_fc_model.ipynb`, and to do the same for the VAE model, use `notebooks/eval_vae_model.ipynb`
