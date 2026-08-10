import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- ТВОЙ КОД МОДЕЛИ ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EMBED = 768
N_HEAD = 12
N_LAYER = 32
BLOCK_SIZE = 2048
VOCAB_SIZE = 50257
MODEL_PATH = "deia_ultra_best.pth"

enc = tiktoken.get_encoding("gpt2")
SPECIAL_TOKENS = {"[USER]", "[AGENT]", "<|endoftext|>"}

def encode(text): return enc.encode(text, allowed_special=SPECIAL_TOKENS)
def decode(tokens): return enc.decode(tokens)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale

def get_rope_freqs(dim, max_len, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(x, freqs):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.size(1), None, :]
    return torch.view_as_real(x_complex * freqs).reshape(*x.shape).type_as(x)

class DeiaBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.wqkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.wo = nn.Linear(n_embed, n_embed, bias=False)
        self.ffn = nn.Sequential(nn.Linear(n_embed, 4 * n_embed, bias=False), nn.GELU(), nn.Linear(4 * n_embed, n_embed, bias=False))
        self.ln1 = RMSNorm(n_embed)
        self.ln2 = RMSNorm(n_embed)

class DeiaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.blocks = nn.ModuleList([DeiaBlock(N_EMBED, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)
        self.token_emb.weight = self.lm_head.weight
        self.register_buffer("freqs", get_rope_freqs(N_EMBED // N_HEAD, 2048, DEVICE))

    def forward(self, idx):
        x = self.token_emb(idx)
        freqs = self.freqs[:idx.size(1)]
        for block in self.blocks:
            h = block.ln1(x)
            q, k, v = block.wqkv(h).chunk(3, dim=-1)
            q = q.view(x.size(0), x.size(1), N_HEAD, -1)
            k = k.view(x.size(0), x.size(1), N_HEAD, -1)
            v = v.view(x.size(0), x.size(1), N_HEAD, -1)
            q, k = apply_rope(q, freqs), apply_rope(k, freqs)
            attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
            x = x + block.wo(attn.contiguous().view(x.size(0), x.size(1), -1))
            x = x + block.ffn(block.ln2(x))
        return self.lm_head(self.ln_f(x))

# Загрузка модели
model = DeiaGPT().to(DEVICE).eval()
if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state"], strict=True)
    print("✅ Model loaded")
else:
    raise FileNotFoundError(MODEL_PATH)

@torch.inference_mode()
def generate_smart(model, idx, max_new_tokens, temperature=0.5, top_p=0.9):
    stop_token_ids = {encode("[USER]")[0], encode("[AGENT]")[0], encode("<|endoftext|>")[0]}
    seq = idx
    kv_cache = None
    for _ in range(max_new_tokens):
        if kv_cache is None: idx_cond = seq[:, -BLOCK_SIZE:]
        else: idx_cond = seq[:, -1:]
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(idx_cond)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
        token_id = int(next_token.item())
        if token_id in stop_token_ids: break
        seq = torch.cat([seq, next_token], dim=1)
    return seq

# --- ВЕБ СЕРВЕР С ОЧЕРЕДЬЮ И НЕБЛОКИРУЮЩИМ САТЙОМ ---
app = FastAPI()

# Блокировка для генерации (гарантирует выполнение строго по очереди)
model_lock = asyncio.Lock()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Отдаёт страницу мгновенно, даже если модель прямо сейчас генерирует ответ другому пользователю
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    prompt = f"[USER] {req.message} [AGENT]"
    tokens = encode(prompt)[-(BLOCK_SIZE - 50):]
    idx = torch.tensor([tokens], device=DEVICE, dtype=torch.long)
    
    # 1. Ждем своей очереди к GPU/модели (если кто-то уже генерирует)
    async with model_lock:
        # 2. Запускаем вычисления в отдельном фоновом потоке, не задерживая сервер
        out = await asyncio.to_thread(generate_smart, model, idx, 2048)

    reply_tokens = out[0, idx.size(1):].tolist()
    reply = decode(reply_tokens).strip()
    
    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)