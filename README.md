
## NMR spectrum translation
Translate NMR spectrum images to SMILES.
This code is for reference only as the dataset is currently not publicly available.

# Dataset:
Our dataset is composed of 135885 simulation NMR spectrums saved as image files.

# Directory structure
- `lib`: tools from   [transformer starter kit](https://www.kaggle.com/c/bms-molecular-translation/discussion/231190)
- `resnet`: contains a Resnet model with it's pretrained weight
- `common`: import necessary libraries
- `configure`: set model parameters
- `dataset`: split the data into training and testing sets
- `model`: combine CNN with Transformer
- `predict`: predict SMILES strings from NMR spectrum
- `train`: train model from training data
- `tokenizer`:  tokenize SMILES from dataset
- `transformer`: modified version of [Transformer](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)



# Model training

This model contains two neural networks. ResNet is used to extract features, which are then fed into the Transformer neural network to predict SMILES. It trained by the following scripts.

	./train.py

Snapshots after each 1000 iteration of training are saved in `result/checkpoint`.  


# SMILES prediction

	./predict.py

result .csv is saved to `result` folder