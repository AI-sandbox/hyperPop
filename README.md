**Hyperbolic Geometry-Based Deep Learning
Methods to Produce Population Trees from
Genotype Data**
==============================


This repo allows for training and evaluation of a variety of models to produce continuous tree representations from genotype data. 
- HypHC Model ([Chami et al. 2020](https://arxiv.org/pdf/2010.00402.pdf)) - use `notebooks/train_simple_model.ipynb` with the appropriate parameter specifications
- MLP (fully connected model) - first, navigate to `hyperLAI/models`. Then, configure the parameters and other specifications in fc_config.json` Finally, run `python train_fc_model.py`
- VAE (variational autoencoder) - first, navigate to `hyperLAI/models`. Then, configure the parameters and other specifications in vae_config.json` Finally, run `python train_vae_model.py`

To evaluate the MLP model and visualize its output, use `notebooks/eval_fc_model.ipynb`, and to do the same for the VAE model, use `notebooks/eval_vae_model.ipynb`
