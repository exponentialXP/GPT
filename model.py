import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from tqdm import tqdm

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
    
def precompute_theta_freqs(args: ModelArgs):
    head_dim = args.emb_dim // args.n_heads
    window_size = args.window_size * 2
    assert head_dim % 2 == 0, "Embedding dimension must be divisible by 2"

    theta_arange = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (args.theta ** (theta_arange / head_dim)).to(args.device)
    m = torch.arange(window_size, device=args.device)
    freqs = torch.outer(m, theta).float()
    complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
    return complex_freqs

def apply_rotary_embeddings(x, freqs_complex, args: ModelArgs):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    freqs_complex = freqs_complex.transpose(1, 2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(args.device)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.head_dim = args.emb_dim // args.n_heads
        self.wq = nn.Linear(args.emb_dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.emb_dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.emb_dim, args.n_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(args.n_heads * self.head_dim, args.emb_dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.residual_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

    def forward(self, x, freqs_complex):
        B, T, C = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.view(B, T, self.args.n_heads, C // self.args.n_heads).transpose(1, 2)
        k = k.view(B, T, self.args.n_heads, C // self.args.n_heads).transpose(1, 2)
        v = v.view(B, T, self.args.n_heads, C // self.args.n_heads).transpose(1, 2)
        q = apply_rotary_embeddings(q, freqs_complex, self.args)
        k = apply_rotary_embeddings(k, freqs_complex, self.args)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.residual_dropout(self.proj(y))

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.emb_dim, args.emb_dim*4, bias=False)
        self.silu = nn.SiLU()
        self.proj = nn.Linear(args.emb_dim*4, args.emb_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.proj(x)  
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm1 = LayerNorm(args.emb_dim)
        self.attn = Attention(args)
        self.norm2 = LayerNorm(args.emb_dim)
        self.mlp = FeedForward(args)
    
    def forward(self, x, freqs_complex):
        x = x + self.attn.forward(self.norm1(x), freqs_complex)
        x = x + self.mlp(self.norm2(x))
        return x

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.emb_dim)
        self.freqs_complex = precompute_theta_freqs(self.args)
        self.dropout = nn.Dropout(args.dropout)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.ln_f = nn.LayerNorm(args.emb_dim)
        self.lm_head = nn.Linear(args.emb_dim, args.vocab_size)

        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
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
        x = self.dropout(self.tok_emb(x))
        freqs_complex = self.freqs_complex[:self.args.window_size] 
        for block in self.blocks:
            x = block(x, freqs_complex)
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
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens=500, temperature=1.0, p=None, view_probabilites=True):
        if view_probabilites == True:
            from sentencepiece import SentencePieceProcessor
            tokenizer_path = 'tokenizer.model'
            tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
            
        for _ in tqdm(range(max_new_tokens), desc="Generating..."):

            x_trim = x[:, -self.args.window_size:]
            logits, _ = self(x_trim)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            if p is not None:
                probs_cumulative = torch.cumsum(probs, dim=0)
                mask = probs_cumulative - probs > p
                probs[mask] = 0.0
                probs.div_(probs.sum(dim=-1, keepdim=True))

            if view_probabilites == True:
                max_displayed_probs = 60
                sorted_probs, indices = torch.sort(probs, descending=True, dim=1)
                for i, (prob, index) in enumerate(zip(sorted_probs[0][:], indices[0][:])):
                    print(f"Token: {tokenizer.Decode(index.tolist())}, Prob: {prob}")
                    if i > max_displayed_probs:
                        print("\n------------------------\n")
                        break
                
            x_next = torch.multinomial(probs, num_samples=1) 
            x = torch.cat((x, x_next), dim=1) 
        return x
