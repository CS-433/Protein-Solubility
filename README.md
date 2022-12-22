# Protein Solubility

The aim of this project is to create a model that can predict solubility
of a protein based on its FASTA sequence.

## How to use

Simply run the script `src/run.py` to run a given model on the dataset,
hyperparameters and models can be configured in the file `src/config.py`

## Structure of source
- `src/run.py`: Main script for training
- `src/train.py`: Helper methods for training
- `src/scores.py`: Methods for evaluating performance of predictions
- `src/models.py`: Defines general architecture of the models used
- `src/config.py`: (Default) Configuration of the models and training
- `src/data.py`: Methods for loading and encoding data

## Jupyter Notebooks

A good part of our work is found in the numerous jupyter notebooks in the root directory:
- `Data_Expl.ipynb`: Contains our initial phases of exploratory data analysis, as well as our attempts at regression analysis, which ended up giving us
better results than all the deep learning models we tried
- `Model#_Eval.ipynb`: Evaluating performance of Model # with given set of parameters by taking the average over several runs
- `Embed_Visualisation.ipynb`: Visualizing the embedding of residues into 2D space that's obtained as a result of training Model 3
- `CNN_Visualisation.ipynb`: Visualizing the output of the first layer of the CNN on random test sequences
- `Regr_Analysis.ipynb`: Contains an attempt at regression analysis by applying PCA to the One Hot representation of the FASTA sequence
