from datasets import load_dataset 
dataset = load_dataset('openwebtext', streaming=True)
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

tokenizer = ByteLevelBPETokenizer()
text = []
examples_to_train = 1_000_000
load_txt_file = False

if load_txt_file == False:
    pbar = tqdm(total=examples_to_train, desc="Processing text")
    for i, example in enumerate(dataset['train']):
        example_text = example['text']
        text.append(example_text)
        del example_text

        if i > examples_to_train:
            break

        pbar.update(1)

    print("Writing text to file...")
    with open('temp.txt', 'w', encoding='utf-8') as f:
        f.writelines(text)

tokenizer.train(['temp.txt'], vocab_size=32_000, min_frequency=2, special_tokens=['<|endoftext|>', '<|$USER|>', '<|$ASSISTANT|>'])
tokenizer.save('tokenizer.json')
