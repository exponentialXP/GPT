from model import Model
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
# torch.manual_seed(42)

save_path = 'modelsave.pt'
tokenizer_path = 'tokenizer.json'

if os.path.exists(save_path):
    checkpoint = torch.load(save_path)
    args = checkpoint['args']
    model = Model(args)
    model.to(device)
    model.load_state_dict(checkpoint['model_params'])
    print(f"Resuming from iter {checkpoint['iter']:,}\nParameters: {sum(p.numel() for p in model.parameters()) - model.pos_emb.weight.numel():,}")

else:
    exit("!!<<Model Checkpoint not found!>>!!")

if os.path.exists(tokenizer_path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    exit("!!<<Tokenizer.json not found!>>!")

context = """: In a shocking finding, scientist discovered a herd of unicorns living
in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the
researchers was the fact that the unicorns spoke perfect English"""
max_new_tokens = 256
p = .9
num_samples = 1
temperature = .8

x = torch.tensor(tokenizer.encode(context).ids, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with torch.amp.autocast(device_type=device, dtype=dtype):
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, p=p, view_probabilites=False)
            print(tokenizer.decode(y[0].tolist()))
            print('---------------')
