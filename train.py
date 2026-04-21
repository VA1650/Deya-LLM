import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import time
import gc
import bitsandbytes as bnb
from torch.utils.checkpoint import checkpoint

# Импортируем архитектуру из model.py
from model import GPT

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
ACCUM_STEPS = 64
BLOCK_SIZE = 1024

D_MODEL = 1024
HEADS = 16
LAYERS = 32

LR = 3e-4
MAX_ITERS = 50000
WARMUP = int(0.02 * MAX_ITERS)
WEIGHT_DECAY = 0.1

DATA_PATH = "input.txt"
LAST_PATH = "last.pt"
EVERY_PATH = "ckpt_{step}.pt"

torch.set_float32_matmul_precision("high")

# ================= TOKENIZER =================
enc = tiktoken.get_encoding("gpt2")
VOCAB = 50257

def encode(t):
    return enc.encode(t, allowed_special={"<|endoftext|>"}, disallowed_special=())

# ================= DATASET =================
class Dataset:
    def __init__(self, path):
        if not os.path.exists(path):
            exit(f"Ошибка: Файл {path} не найден!")
        with open(path, "r", encoding="utf-8") as f:
            self.data = torch.tensor(encode(f.read()), dtype=torch.long)
        self.n = len(self.data)

    def get(self):
        ix = torch.randint(0, self.n - BLOCK_SIZE - 1, (BATCH_SIZE,))
        x = torch.stack([self.data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([self.data[i+1:i+BLOCK_SIZE+1] for i in ix])
        return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

# ================= LR SCHEDULER =================
def get_lr(step):
    if step < WARMUP:
        return LR * (step / WARMUP)
    
    progress = (step - WARMUP) / (MAX_ITERS - WARMUP)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    
    min_lr = LR * 0.1
    return min_lr + (LR - min_lr) * cosine

# ================= INITIALIZATION =================
model = GPT(VOCAB, D_MODEL, HEADS, LAYERS, BLOCK_SIZE).to(DEVICE)

# Используем 8-битную оптимизацию для экономии памяти
opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler()

start_step = 0

# Загрузка последнего чекпоинта
if os.path.exists(LAST_PATH):
    print(f"Загрузка прогресс из {LAST_PATH}...")
    ckpt = torch.load(LAST_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['opt'])
    start_step = ckpt['step'] + 1
    print(f"✅ Продолжаем с шага {start_step}")
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()

data = Dataset(DATA_PATH)

# ================= TRAINING LOOP =================
print(f"🚀 Старт обучения на {DEVICE}...")

for step in range(start_step, MAX_ITERS):
    t0 = time.time()
    
    # Обновляем LR
    current_lr = get_lr(step)
    for g in opt.param_groups:
        g["lr"] = current_lr

    opt.zero_grad(set_to_none=True)
    total_loss = 0

    # Накопление градиента 
    for _ in range(ACCUM_STEPS):
        x, y = data.get()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
            loss = loss / ACCUM_STEPS
        
        scaler.scale(loss).backward()
        total_loss += loss.item()

    # Степ оптимизатора
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(opt)
    scaler.update()

    # Замер времени
    torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000

    if step % 10 == 0:
        print(f"step {step} | loss {total_loss:.4f} | lr {current_lr:.2e} | {dt:.0f}ms")

    # Сохранение
    if step > 0 and step % 50 == 0:
        torch.save({
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'step': step,
        }, LAST_PATH)
        
    if step % 1000 == 0 and step > 0:
        torch.save(model.state_dict(), EVERY_PATH.format(step=step))
        print(f"Сохранен чекпоинт шага {step}")

    # Очистка памяти
    if step % 100 == 0:
        gc.collect()
        torch.cuda.empty_cache()
