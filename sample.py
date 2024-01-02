from model import Model
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# torch.manual_seed(42)

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
max_new_tokens = 256
top_k = 200 # See if setting this back to 200 helps 
num_samples = 1
temperature = 0.8 

x = torch.tensor(tokenizer.encode(context).ids, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(tokenizer.decode(y[0].tolist()))
            print('---------------')
