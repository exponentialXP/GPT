import tracemalloc
import pickle
import os
from datasets import load_dataset 
from tokenizers import Tokenizer
import numpy as np

if __name__ == '__main__':
    arr_data = []
    val_data = []
    
    max_examples = 50_000
    max_val_examples = 5000
    tracemalloc.start()
    dataset = load_dataset('openwebtext', streaming=True)

    tokenizer = Tokenizer.from_file('tokenizer.json')

    with open('data.pkl', 'wb') as f, open('val_data.pkl', 'wb') as f_val:
        examples = 0
        for example in dataset['train']:
            data = tokenizer.encode(example['text']).ids + tokenizer.encode("<|endoftext|>").ids
            if examples < max_examples:
                arr_data.append(data)
            elif examples <= max_examples+max_val_examples:
                val_data.append(data)
            else:
                narr_data = np.concatenate(arr_data)
                if val_data:
                    nval_data = np.concatenate(val_data)
                else:
                    exit("!!<<No Validation Data>>!!")

                pickle.dump(narr_data, f)
                pickle.dump(nval_data, f_val)
                break

            del data
            
            examples += 1
            if examples % 1000 == 0:
                print(f"Examples Processed: {examples}")

    print("Current: %d, Peak %d" % tracemalloc.get_traced_memory())
    print(f"Current data.pkl size: {os.path.getsize('data.pkl')/(1024*1024):.3f}MB | Current val data.pkl size: {os.path.getsize('val_data.pkl')/(1024*1024):.3f}MB")

    with open('data.pkl', 'rb') as f:
        arr_data = []
        while True:
            try:
                data = pickle.load(f)
                arr_data.append(data)
            except EOFError:
                break
    
    with open('val_data.pkl', 'rb') as f_val:
        arr_val_data = []
        while True:
            try:
                data_val = pickle.load(f_val)
                arr_val_data.append(data_val)
            except EOFError:
                break

    # print(f"-----------------Train\n{arr_data}\n ------------------Val\n{arr_val_data}")
