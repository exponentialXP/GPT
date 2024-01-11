import torch
import torch.nn as nn
from torch.nn import functional as F
from model import Model, ModelArgs
import time
import math
import numpy as np
from contextlib import nullcontext

# Decrease emb_dim, n_heads, n_layers and batch_size if running out of memory

load = True # True to load model checkpoint, False to not load. WARNING: If False, it can override checkpoints!
batch_size = 8
gradient_accumulation_steps = 40
grad_clip = 1.0
window_size = 512
emb_dim = 512 
n_heads = 8
n_layers = 8
max_iters = 10_000
eval_interval = 100
save_interval = 100
warmup_iters = 300
lr_decay_iters = max_iters
learning_rate = 6e-4
min_lr = 6e-5
dropout = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
fused = True if torch.cuda.is_available() else False
eval_iters = 200

modelsave_path = 'modelsave.pt'
tokenizer_path = 'tokenizer.model'

torch.manual_seed(42)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=dtype)

train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
print(f"Amount of tokens in training dataset: {train_data.shape[0]:,}")

import os
if os.path.exists(tokenizer_path):
    from sentencepiece import SentencePieceProcessor
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
    vocab_size = tokenizer.vocab_size()
else:
    exit("!!<<No tokenizer found>>!!")

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
            xv, yv = get_batch(split)
            logits, loss = model(xv, yv)
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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1, betas=(0.9, 0.95), fused=True)

if os.path.exists(modelsave_path) and load == True:
    checkpoint = torch.load(modelsave_path)
    model.load_state_dict(checkpoint['model_params'])
    optimizer.load_state_dict(checkpoint['optimizer_params'])
    iter = checkpoint['iter']
    print(f"Loaded checkpoint. Starting from iter {checkpoint['iter']:,}")
else:
    iter = 0
    print("?<No model checkpoint, training from scratch>?")

xb, yb = get_batch('train')
start_time = time.time()
while iter < max_iters:
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(iter)
    
    if iter % save_interval == 0 and iter != 0:
        checkpoint = {
            'model_params': model.state_dict(), 
            'optimizer_params': optimizer.state_dict(), 
            'args': args,
            'iter': iter
        }
        torch.save(checkpoint, modelsave_path)
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        end_time = time.time() 
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time: {end_time - start_time:.4f}s")
        start_time = time.time()

    for g in range(gradient_accumulation_steps):
        with torch.amp.autocast(device_type=device, dtype=dtype):
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps

        xb, yb = get_batch('train')
        scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    iter += 1

context = torch.tensor((tokenizer.encode("<|endoftext|>").ids), dtype=torch.long, device=device).unsqueeze(0)
print(tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
