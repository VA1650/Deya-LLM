import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import gc
import time
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# --- КОНФИГУРАЦИЯ "DEIA-STABLE-LOCAL" ---
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8 
BLOCK_SIZE = 1024 
MAX_ITERS = 50000
LEARNING_RATE = 1e-4    
DEVICE = 'cuda'
N_EMBED = 768
N_HEAD = 12
N_LAYER = 32
DROPOUT = 0.1

INPUT_FILE = 'input.txt'
SAVE_PATH = 'deia_ultra.pth'
BEST_PATH = 'deia_ultra_best.pth'

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- 1. ТОКЕНИЗАТОР ---
enc = tiktoken.get_encoding("gpt2")
SPECIAL_TOKENS = {"[USER]", "[AGENT]", "<|endoftext|>"}
def encode(text): return enc.encode(text, allowed_special=SPECIAL_TOKENS, disallowed_special=())
def decode(tokens): return enc.decode(tokens)
vocab_size = 50257

# --- 2. АРХИТЕКТУРА ---
def get_rope_freqs(dim, max_len, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device)
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rope(x, freqs):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.size(1), None, :]
    return torch.view_as_real(x_complex * freqs).reshape(*x.shape).type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale

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
        self.ln1, self.ln2 = RMSNorm(n_embed), RMSNorm(n_embed)

    def forward(self, x, freqs):
        h = self.ln1(x)
        q, k, v = self.wqkv(h).chunk(3, dim=-1)
        q = q.view(x.size(0), x.size(1), self.n_head, self.head_size)
        k = k.view(x.size(0), x.size(1), self.n_head, self.head_size)
        v = v.view(x.size(0), x.size(1), self.n_head, self.head_size)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, dropout_p=DROPOUT if self.training else 0
        ).transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        x = x + self.wo(out)
        x = x + self.ffn(self.ln2(x))
        return x

class DeiaGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, N_EMBED)
        self.blocks = nn.ModuleList([DeiaBlock(N_EMBED, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = RMSNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight
        self.register_buffer("freqs", get_rope_freqs(N_EMBED // N_HEAD, 2048, DEVICE))

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        freqs = self.freqs[:idx.size(1)]
        for block in self.blocks:
            x = gradient_checkpoint(block, x, freqs, use_reentrant=False)
        logits = self.lm_head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            # Жестко ограничиваем вход, чтобы не раздувать память
            idx_cond = idx[:, -BLOCK_SIZE:] 
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # Чистим промежуточные логиты
            del logits, probs
        self.train()
        return idx

# --- 3. ДАННЫЕ ---
print("Токенизация данных...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f: text_data = f.read()
full_data = torch.tensor(encode(text_data), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(full_data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([full_data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([full_data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# --- 4. ЗАПУСК ---
model = DeiaGPT(vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
scaler = torch.amp.GradScaler('cuda')
start_iter = 0
best_loss = float('inf')

if os.path.exists(BEST_PATH):
    print(f"Загрузка из {BEST_PATH}...")
    ckpt = torch.load(BEST_PATH, map_location=DEVICE)
    if ckpt.get('loss', 100) < 40.0: # Подняли планку, т.к. начали с 80
        model.load_state_dict(ckpt['state'])
        optimizer.load_state_dict(ckpt['opt'])
        start_iter = ckpt.get('iter', 0)
        best_loss = ckpt['loss']
        print(f"✅ Восстановлено! Шаг: {start_iter}, Loss: {best_loss:.4f}")

print(f"Параметров: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
iter_start_time = time.time()

# --- ЦИКЛ ---
for iter in range(start_iter, MAX_ITERS):
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0
    
    for _ in range(ACCUMULATION_STEPS):
        xb, yb = get_batch()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(xb, yb)
            loss = loss / ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        loss_accum += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if iter % 50 == 0:
        current_loss = loss_accum * ACCUMULATION_STEPS
        t_end = time.time()
        speed = (t_end - iter_start_time)
        print(f"\n--- Шаг {iter} | Loss: {current_loss:.4f} | Время 50 ит: {speed:.1f}с ---")
        
        seed = torch.tensor([encode("[USER] Привет")], device=DEVICE)
        torch.cuda.empty_cache()
        gc.collect()
        out = model.generate(seed, max_new_tokens=60)[0].tolist()
        print(f"ДЕЯ: {decode(out)}")
        
        if current_loss < 40.0: # Сохраняем, если не взорвалось
            ckpt = {'state': model.state_dict(), 'opt': optimizer.state_dict(), 'iter': iter, 'loss': current_loss}
            torch.save(ckpt, SAVE_PATH)
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(ckpt, BEST_PATH)
                print("✨ Рекорд!")
        
        iter_start_time = time.time()

        torch.cuda.empty_cache()
        gc.collect()
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        #print(torch.cuda.memory_summary(device=DEVICE, abbreviated=True))

    if iter > 0 and iter % 1000 == 0:
        torch.save(ckpt, f'deia_backup_{iter}.pth')
        
