from model import Model
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
# torch.manual_seed(42)

if os.path.exists('modelsave.pt'):
    checkpoint = torch.load('modelsave.pt')
    args = checkpoint['args']
    model = Model(args)
    model.to(device)
    model.load_state_dict(checkpoint['model_params'])
else:
    exit("!!<<modelsave.pt (Model Checkpoint) not found!>>!!")

print(f"Resuming from iter {checkpoint['iter']:,}\nParameters: {sum(p.numel() for p in model.parameters()) - model.pos_emb.weight.numel():,}")
if os.path.exists('tokenizer.json'):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('tokenizer.json')
else:
    exit("!!<<Tokenizer.json not found!>>!")

context = """<|endoftext|>""" # What are the next token(s) the model should predict based off the context?
max_new_tokens = 256
p = .9
num_samples = 1
temperature = 1

x = torch.tensor(tokenizer.encode(context).ids, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with torch.amp.autocast(device_type=device, dtype=dtype):
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, p=p)
            print(tokenizer.decode(y[0].tolist()))
            print('---------------')
