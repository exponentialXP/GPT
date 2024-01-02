# CREATE YOUR TOKENIZER HERE
from datasets import load_dataset 
dataset = load_dataset('openwebtext', streaming=True)
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
text = []
for i, example in enumerate(dataset['train']):
    example_text = example['text']
    text.append(example_text)
    del example_text

    if i > 50000:
        break

with open('temp.txt', 'w', encoding='utf-8') as f:
    f.writelines(text)

tokenizer.train(['temp.txt'], vocab_size=30000, min_frequency=2, special_tokens=['<|endoftext|>', '<|$USER|>', '<|$ASSISTANT|>'])
tokenizer.save('tokenizer.json')
