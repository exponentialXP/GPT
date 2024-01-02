import torch
import torch.nn as nn
from torch.nn import functional as F
from model import Model, ModelArgs
import time
import math
import pickle
import numpy as np

load = True
batch_size = 32
gradient_accumulation_steps = 6
grad_clip = 1.0
window_size = 256 
emb_dim = 128
n_heads = 4
n_layers = 4
max_iters = 3_000
eval_interval = 100
save_interval = 300
warmup_iters = 300
lr_decay_iters = max_iters
learning_rate = 6e-4
min_lr = 6e-5
dropout = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(42)

scaler = torch.cuda.amp.GradScaler()

train_data, val_data = pickle.load(open('data.pkl', 'rb')), pickle.load(open('val_data.pkl', 'rb'))

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("./tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

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
    x = torch.stack([torch.from_numpy((data[i:i+window_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+window_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
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

    for g in range(gradient_accumulation_steps):
        xb, yb = get_batch('train')
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    iter += 1

context = torch.tensor((tokenizer.encode("[EOS]").ids), dtype=torch.long, device=device).unsqueeze(0)
print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
