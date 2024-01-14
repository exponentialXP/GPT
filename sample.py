from model import Model
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

seed = None # None to make it random every time!
mode = 'print' # write to write to file, print to print text
write_file_path = 'generation.txt'

if seed is not None:
    torch.manual_seed(42)

save_path = 'modelsave.pt'
tokenizer_path = 'tokenizer.json'

if os.path.exists(save_path):
    checkpoint = torch.load(save_path)
    args = checkpoint['args']
    model = Model(args)
    model.to(device)
    model.load_state_dict(checkpoint['model_params'])
    print(f"Resuming from iter {checkpoint['iter']:,}\nParameters: {model.count_params():,}")

else:
    exit("!!<<Model Checkpoint not found!>>!!")

import os
if os.path.exists(tokenizer_path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    exit("!!<<No tokenizer found>>!!")

context = "<|endoftext|>"
max_new_tokens = 1500
p = .9
num_samples = 999999
temperature = .95
view_probabilites = False

x = torch.tensor(tokenizer.encode(context).ids, dtype=torch.long, device=device).unsqueeze(0)

if mode == 'print':
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=dtype):
            for k in range(num_samples):
                print("\n", context, end="")
                model.generate(x, max_new_tokens, temperature=temperature, p=p, view_probabilites=view_probabilites)
                print('\n---------------\n')

elif mode == 'write':
    with open(write_file_path, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=dtype):
                for k in range(num_samples):
                    y = model.generate(x, max_new_tokens, mode='write', temperature=temperature, p=p, view_probabilites=view_probabilites)
                    f.write(tokenizer.decode(y[0].tolist())) 
                    f.write('\n---------------\n')
