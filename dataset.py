from datasets import load_dataset
from tqdm import tqdm

max_examples = 1_000_000 # 1 example = 1,000 tokens
dset = load_dataset("openwebtext", split=f'train[0:{max_examples}]', cache_dir='./cache')

split_dataset = dset.train_test_split(test_size=0.05, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

from sentencepiece import SentencePieceProcessor
tokenizer_path = 'tokenizer.model'
tokenizer = SentencePieceProcessor(model_file=tokenizer_path)

eot_id = tokenizer.Encode('<|endoftext|>')

def process(example):
    tokens = tokenizer.Encode(example['text']) + eot_id
    return {'tokens': tokens, 'len': len(tokens)}

tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc='Tokenizing the splits',
)

import numpy as np

for split, example in tokenized.items():
    length = np.sum(example['len'], dtype=np.int64)
    filename = f"{split}.bin"
    map = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(length,))

    total_batches = 128
    start_idx = 0

    for ix in tqdm(range(total_batches), desc=f'Writing {filename}'):
        batch = example.shard(num_shards=total_batches, index=ix, contiguous=True).with_format('numpy')
        map_batch = np.concatenate(batch['tokens'])
        map[start_idx: start_idx + len(map_batch)] = map_batch
        start_idx += len(map_batch)

    map.flush()
