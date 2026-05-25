import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

# ================= ROPE =================

def build_rope(head_dim, max_len, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()[None, :, None, :] 
    sin = freqs.sin()[None, :, None, :]
    return cos, sin

def apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    T = x.shape[1]
    c = cos[:, :T, :]
    s = sin[:, :T, :]
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.stack((out1, out2), dim=-1).flatten(-2)

# ================= RMS =================

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.g

# ================= BLOCK =================

class Block(nn.Module):
    def __init__(self, d, h, dropout=0.1):
        super().__init__()
        self.h = h
        self.dh = d // h

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d)
        )
        self.resid_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

        self.ln1 = RMSNorm(d)
        self.ln2 = RMSNorm(d)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)

        q = q.view(B, T, self.h, self.dh)
        k = k.view(B, T, self.h, self.dh)
        v = v.view(B, T, self.h, self.dh)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=0.1 if self.training else 0.0
        )

        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.resid_drop(self.proj(attn))
        x = x + self.ffn_drop(self.ffn(self.ln2(x)))
        return x

# ================= MODEL =================

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, heads, layers, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, heads, dropout=0.1)
            for _ in range(layers)
        ])

        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.emb.weight

        cos, sin = build_rope(d_model // heads, 4096, "cpu") 
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight') or pn.endswith('ffn.2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        x = self.emb(x)
        cos, sin = self.cos, self.sin

        for b in self.blocks:
            if self.training:
                x = checkpoint(b, x, cos, sin, use_reentrant=False)
            else:
                x = b(x, cos, sin)

        logits = self.head(self.norm(x))

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))

        return logits, loss
