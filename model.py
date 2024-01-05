import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

@dataclass
class ModelArgs:
    emb_dim: int = -1
    n_heads: int = -1
    n_layers: int = -1
    dropout: float = -1
    window_size: int = -1
    vocab_size: int = -1
    device: str = 'cpu'

class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.head_dim = args.emb_dim // args.n_heads
        self.wq = nn.Linear(args.emb_dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.emb_dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.emb_dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.emb_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(args.window_size, args.window_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        v = self.wv(x)
        out = wei @ v
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.emb_dim, args.emb_dim*4, bias=False)
        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(args.emb_dim*4, args.emb_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.c_proj(x)  
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm1 = LayerNorm(args.emb_dim)
        self.attn = Attention(args)
        self.norm2 = LayerNorm(args.emb_dim)
        self.mlp = FeedForward(args)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.emb_dim)
        self.pos_emb = nn.Embedding(args.window_size, args.emb_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.ln_f = nn.LayerNorm(args.emb_dim)
        self.lm_head = nn.Linear(args.emb_dim, args.vocab_size)

        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok = self.tok_emb(x) 
        pos = self.pos_emb(torch.arange(T, device=self.args.device))
        x = self.dropout(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens=500, temperature=1.0, p=None):
        for _ in range(max_new_tokens):
            x_trim = x[:, -self.args.window_size:]
            logits, _ = self(x_trim)
            logits = logits[:, -1, :] / temperature
            if p is not None:
                probs, indices = torch.sort(F.softmax(logits, dim=-1), descending=True)
                probs_cumulative = torch.cumsum(probs, dim=0)
                mask = probs_cumulative - probs > p
                probs[mask] = 0.0
                probs.div_(probs.sum(dim=-1, keepdim=True))
            # max_displayed_probs = 60
            # from tokenizers import Tokenizer
            # tokenizer = Tokenizer.from_file('./tokenizer.json')
            # for i, (prob, index) in enumerate(zip(probs[0][:], indices[0][:])):
            #     print(f"Token: {tokenizer.id_to_token(index)}, Prob: {prob}")
            #     if i > max_displayed_probs or prob == 0:
            #         break
            x_next = torch.multinomial(probs, num_samples=1) 
            x = torch.cat((x, x_next), dim=1) 
        return x
