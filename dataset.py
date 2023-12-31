# CURRENTLY NOT USABLE FOR HUGE (like 8+ GB or whatever is your RAM) DATASETS!

import torch
from datasets import load_dataset

dataset = load_dataset('Skylion007/openwebtext', streaming=True)
max_chars = 100_000_000
max_val_chars = 10_000_000
output_location = 'raw-input.txt'
val_output_location = 'raw-val-input.txt'

num_chars = 0
with open(output_location, 'w', encoding='utf-8') as train_file, open(val_output_location, 'w', encoding='utf-8') as val_file:
    for example_chunk in dataset['train']:
        for example in example_chunk['text']:

            if num_chars % 1_000_000 == 0:
                print(f"{num_chars/1_000_000}MB written")

            if num_chars < max_chars:
                    train_file.write(example)
                    num_chars += 1

            elif num_chars < max_chars + max_val_chars:
                    val_file.write(example)
                    num_chars += 1
            else:
                break
        
        if num_chars >= max_chars + max_val_chars:
            break
            
        
with open(output_location, 'r', encoding='utf-8') as f:
    train_text = f.read()

with open(val_output_location, 'r', encoding='utf-8') as f:
    val_text = f.read()

from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train([output_location, val_output_location], vocab_size=12000)
tokenizer.save('tokenizer.json')

tokenized_train_text = tokenizer.encode(train_text).ids
tokenized_val_text = tokenizer.encode(val_text).ids
torch.save(tokenized_train_text, 'tokenized_train_text.pt')
torch.save(tokenized_val_text, 'tokenized_val_text.pt')
