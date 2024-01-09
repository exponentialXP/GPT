Requirements:
torch, numpy, tokenizers

GPU not needed, but will speed up TRAIN.py. 
Feel free to experiment with the hyperparameters!

Guide:
1. Use TOKENIZERTRAIN.py to train the tokenizer on the dataset (do this first) 2. Use DATASET.py to download and tokenize the dataset

Use TRAIN.py to download the model on your dataset

Use SAMPLE.py to generate using your model checkpoint

Happy Language Modelling! :)

Note: A good chunk of the MODEL.py and TRAIN.py's code was from Andrej Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
