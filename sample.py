from model import Model
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if os.path.exists('modelsave.pt'):
    checkpoint = torch.load('modelsave.pt')
    args = checkpoint['args']
    model = Model(args)
    model.to(device)
    model.load_state_dict(checkpoint['model_params'])
else:
    exit("!!<<modelsave.pt (Model Checkpoint) not found!>>!!")

if os.path.exists('tokenizer.json'):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('tokenizer.json')
else:
    exit("!!<<Tokenizer.json not found!>>!")

model.eval()
context = "<|endoftext|>"
max_new_tokens = args.window_size 
num_samples = 5

x = torch.tensor(tokenizer.encode(context).ids, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens)
            print(tokenizer.decode(y[0].tolist()))
            print('---------------')
