import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- НАСТРОЙКИ ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "deia_checkpoint.pth"  # или deia_checkpoint.pth
TOKENIZER_PATH = "tokenizer.json"

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Не найден {TOKENIZER_PATH}")

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()

def encode(text): 
    return tokenizer.encode(text).ids

def decode(tokens): 
    return tokenizer.decode(tokens, skip_special_tokens=False)

# Спецтокены
USER_ID = tokenizer.token_to_id("[USER]")
AGENT_ID = tokenizer.token_to_id("[AGENT]")
EOT_ID = tokenizer.token_to_id("[EOT]")
EOS_ID = tokenizer.token_to_id("[EOS]")

# --- МОДЕЛЬ ---
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
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed, bias=False)
        )
        self.ln1 = RMSNorm(n_embed)
        self.ln2 = RMSNorm(n_embed)

    def forward(self, x, freqs):
        h = self.ln1(x)
        q, k, v = self.wqkv(h).chunk(3, dim=-1)
        B, T, C = x.shape
        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2).contiguous().view(B, T, C)
        
        x = x + self.wo(out)
        x = x + self.ffn(self.ln2(x))
        return x

class DeiaGPT(nn.Module):
    def __init__(self, vocab_size, n_embed=768, n_head=12, n_layer=32, block_size=1024):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([DeiaBlock(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
        self.register_buffer("freqs", get_rope_freqs(n_embed // n_head, block_size, "cpu"))

    def forward(self, idx):
        x = self.token_emb(idx)
        freqs = self.freqs[:idx.size(1)]
        for block in self.blocks:
            x = block(x, freqs)
        x = self.ln_f(x)
        return self.lm_head(x)

# Загрузка Чекпоинта и Модели
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Файл чекпоинта не найден: {MODEL_PATH}")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

N_EMBED = ckpt.get("n_embed", 768)
N_HEAD = ckpt.get("n_head", 12)
N_LAYER = ckpt.get("n_layer", 32)
BLOCK_SIZE = ckpt.get("block_size", 1024)

model = DeiaGPT(
    vocab_size=VOCAB_SIZE, 
    n_embed=N_EMBED, 
    n_head=N_HEAD, 
    n_layer=N_LAYER, 
    block_size=BLOCK_SIZE
).to(DEVICE).eval()

model.load_state_dict(ckpt["state"], strict=True)
print("✅ Новая модель успешно загружена!")

# --- ГЕНЕРАЦИЯ ---
@torch.inference_mode()
def generate_smart(model, idx, max_new_tokens, temperature=0.7, top_k=50):
    # Допустимые токены остановки
    stop_tokens = {e for e in [EOT_ID, EOS_ID, USER_ID] if e is not None}
    
    seq = idx
    for _ in range(max_new_tokens):
        idx_cond = seq[:, -BLOCK_SIZE:]
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            logits = model(idx_cond)[:, -1, :]
            
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        token_id = int(next_token.item())
        if token_id in stop_tokens:
            break
            
        seq = torch.cat([seq, next_token], dim=1)
        
    return seq

# --- API СЕРВЕР ---
app = FastAPI()
model_lock = asyncio.Lock()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Server is running</h1>"

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    prompt = f"[USER] {req.message}\n[AGENT] "
    tokens = encode(prompt)[-(BLOCK_SIZE - 50):]
    idx = torch.tensor([tokens], device=DEVICE, dtype=torch.long)
    
    async with model_lock:
        out = await asyncio.to_thread(generate_smart, model, idx, 512, 0.7, 50)

    reply_tokens = out[0, idx.size(1):].tolist()
    reply = decode(reply_tokens)
    
    # Очистка служебных тегов из ответа
    for tag in ["[USER]", "[AGENT]", "[EOT]", "[EOS]"]:
        if tag in reply:
            reply = reply.split(tag)[0]
            
    return {"reply": reply.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)