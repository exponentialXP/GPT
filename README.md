This repository is all for speed + performance while keeping the code simple and easy to read

Requirements:
torch 2.0+, numpy, tokenizers 
Requirements Install Tutorial (super easy!)
(in command prompt/terminal do pip install numpy tokenizers, and then go to https://pytorch.org and put in your OS, package as pip, language as python and pick the latest stable version and put in command prompt/terminal

GPU not needed, but will speed up TRAIN.py. 
To use CUDA to GREATLY speed up training follow this tutorial :): https://www.youtube.com/watch?v=r7Am-ZGMef8&t=542s
Feel free to experiment with the hyperparameters!

Guide:
1. Use TOKENIZERTRAIN.py to train the tokenizer on the dataset (do this first) 2. Use DATASET.py to download and tokenize the dataset

Use TRAIN.py to download the model on your dataset

Use SAMPLE.py to generate using your model checkpoint

Happy Language Modelling! :)

Note: A good chunk of the MODEL.py and TRAIN.py's code was from Andrej Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
