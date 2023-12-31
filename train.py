import torch
import torch.nn as nn
from torch.nn import functional as F
from model import Model, ModelArgs
import time
import math
import numpy as np

load = True
batch_size = 64 
window_size = 256 
emb_dim = 256
n_heads = 4
n_layers = 4
max_iters = 2_000
eval_interval = 300
save_interval = 300
warmup_iters = 300
lr_decay_iters = max_iters
learning_rate = 6e-4
min_lr = 6e-5
dropout = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
vocab_size = 12000

torch.manual_seed(42)

train_data, val_data = torch.tensor(torch.load('tokenized_train_text.pt'), dtype=torch.long), torch.tensor(torch.load('tokenized_val_text.pt'), dtype=torch.long)
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("./tokenizer.json")

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - window_size, (batch_size,))
    x = torch.stack([data[i:i+window_size] for i in ix])
    y = torch.stack([data[i+1:i+window_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

args = ModelArgs(emb_dim=emb_dim, n_heads=n_heads, 
                window_size=window_size, n_layers=n_layers,
                dropout=dropout,
                vocab_size=vocab_size, device=device)

model = Model(args).to(device)
print(f"Number of parameters: {model.count_params():,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1, betas=(0.9, 0.95))

import os
if os.path.exists('modelsave.pt') and load == True:
    checkpoint = torch.load('modelsave.pt')
    model.load_state_dict(checkpoint['model_params'])
    optimizer.load_state_dict(checkpoint['optimizer_params'])
    iter = checkpoint['iter']
else:
    iter = 0

start_time = time.time()
while iter < max_iters:
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(iter)

    if iter % eval_interval == 0:
        losses = estimate_loss()
        end_time = time.time() 
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time: {end_time - start_time:.4f}s")
        start_time = time.time()
    
    if iter % save_interval == 0:
        checkpoint = {
            'model_params': model.state_dict(), 
            'optimizer_params': optimizer.state_dict(), 
            'args': args,
            'iter': iter
        }
        torch.save(checkpoint, f'modelsave.pt')

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    iter += 1

context = torch.tensor((tokenizer.encode("Third Citizen:").ids), dtype=torch.long, device=device).unsqueeze(0)
print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
