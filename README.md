# Protein Solubility

EPFL Machine Learning Course, Autumn 2022 - Class Project 1

Team members: Lubor Budaj, Matthew Dupraz, Anton Hosgood

This repository contains the code produced during this project. The aim of the project is to create a model that can classify a protein as either soluble or insoluble based only on its FASTA sequence.

## How To Use

Simply run the script `src/run.py` to run a given model on the dataset. Hyperparameters and models can be configured in `src/config.py`.

## Overview

- `data/` - datasets containing FASTA sequences and labels denoting solubility

- `src/` - source code used in our pipeline

The rest of the root directory contains notebooks holding analyses and other work carried out.

## Source Structure

- `src/config.py` - (default) configuration of the models and training
- `src/data.py` - methods for loading and encoding data
- `src/models.py` - defines general architectures of the models used
- `src/scores.py` - methods for evaluating model performance
- `src/train.py` - helper methods for training
- `src/run.py` - main script for training

## Jupyter Notebooks

A good part of our work is found in the numerous Hupyter notebooks in the root directory:

- `Data_Expl.ipynb` - contains the initial phases of our exploratory data analysis, as well as our attempts at regression analysis, which ended up giving us better results than the deep learning models
- `Model#_Eval.ipynb` - evaluating performance of model # with given set of parameters by taking the average over several runs
- `Embed_Visualisation.ipynb` - visualising the embedding of residues into 2D space that is obtained as a result of training model 3
- `CNN_Visualisation.ipynb` - visualising the output of the first layer of the CNN on random test sequences
- `Regr_Analysis.ipynb` - contains an attempt at regression analysis by applying PCA to the one-hot representation of the FASTA sequence

# Environment

We use `Python 3.9.12` and `PyTorch` to build our deep learning models. Several other libraries are used including `NumPy`, `pandas`, `scikit-learn`.

`Matplotlib` and `seaborn` are used for visualisation purposes.
