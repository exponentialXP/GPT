# CURRENTLY NOT USABLE FOR HUGE (like 8+ GB or whatever is your RAM) DATASETS!

import torch
from datasets import load_dataset

dataset = load_dataset('Skylion007/openwebtext', streaming=True)
max_examples = 10_000
max_val_examples = 500
print_frequency = 100
output_location = 'raw-input.txt'
val_output_location = 'raw-val-input.txt'

num_examples = 0
with open(output_location, 'w', encoding='utf-8') as train_file, open(val_output_location, 'w', encoding='utf-8') as val_file:
    for example_chunk in dataset['train']:
        example = example_chunk['text']
        if num_examples % print_frequency == 0:
            print(f"{num_examples:,} examples written")

        if num_examples < max_examples:
            train_file.write(f"{example}")
            num_examples += 1

        elif num_examples < max_examples + max_val_examples:
            val_file.write(f"{example}")
            num_examples += 1
        else:
            break
        
        if num_examples >= max_examples + max_val_examples:
            break
            
        
with open(output_location, 'r', encoding='utf-8') as f:
    train_text = f.read()

with open(val_output_location, 'r', encoding='utf-8') as f:
    val_text = f.read()

from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train([output_location, val_output_location], vocab_size=12000)
tokenizer.save('tokenizer.json')

print("Tokenizing...")
tokenized_train_text = tokenizer.encode(train_text).ids
tokenized_val_text = tokenizer.encode(val_text).ids
torch.save(tokenized_train_text, 'tokenized_train_text.pt')
torch.save(tokenized_val_text, 'tokenized_val_text.pt')
