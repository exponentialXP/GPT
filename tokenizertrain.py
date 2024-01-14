from datasets import load_dataset 
dataset = load_dataset('openwebtext', cache_dir='cache')
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import os

tokenizer = ByteLevelBPETokenizer()

examples_to_train = 200_000

pbar = tqdm(total=examples_to_train, desc="Processing...")

with open('temp.txt', 'a', encoding='utf-8') as f:
    for i, example in enumerate(dataset['train']):
        if i < examples_to_train:
            example_text = example['text']
            f.write(example_text)
            del example_text
            pbar.update(1)
        else:
            break

tokenizer.train(['temp.txt'], vocab_size=32_000, min_frequency=2, special_tokens=['<|endoftext|>', '<|$USER|>', '<|$ASSISTANT|>'])
os.remove('temp.txt')
tokenizer.save('tokenizer.json')
