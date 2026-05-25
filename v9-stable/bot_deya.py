import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import asyncio

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.chat_action import ChatActionSender

# ---------------- CONFIG ----------------
TOKEN = 'you token'
ADMIN_IDS = 'you ids'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EMBED = 768
N_HEAD = 12
N_LAYER = 32
BLOCK_SIZE = 1024
VOCAB_SIZE = 50257
MODEL_PATH = "deia_ultra_best.pth"

# ---------------- TOKENIZER ----------------
enc = tiktoken.get_encoding("gpt2")

SPECIAL_TOKENS = {"[USER]", "[AGENT]", "<|endoftext|>"}

def encode(text):
    return enc.encode(text, allowed_special=SPECIAL_TOKENS)

def decode(tokens):
    return enc.decode(tokens)

# ---------------- MODEL (UNCHANGED) ----------------
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

        q = q.view(x.size(0), x.size(1), self.n_head, self.head_size)
        k = k.view(x.size(0), x.size(1), self.n_head, self.head_size)
        v = v.view(x.size(0), x.size(1), self.n_head, self.head_size)

        q, k = apply_rope(q, freqs), apply_rope(k, freqs)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True
        ).transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)

        x = x + self.wo(out)
        x = x + self.ffn(self.ln2(x))
        return x


class DeiaGPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(VOCAB_SIZE, N_EMBED)

        self.blocks = nn.ModuleList([
            DeiaBlock(N_EMBED, N_HEAD) for _ in range(N_LAYER)
        ])

        self.ln_f = RMSNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE, bias=False)

        self.token_emb.weight = self.lm_head.weight

        self.register_buffer(
            "freqs",
            get_rope_freqs(N_EMBED // N_HEAD, 2048, DEVICE)
        )

        # ---------------- KV CACHE ----------------
        self.k_cache = [None] * N_LAYER
        self.v_cache = [None] * N_LAYER

    def forward(self, idx):
        x = self.token_emb(idx)
        freqs = self.freqs[:idx.size(1)]

        for block in self.blocks:
            x = block(x, freqs)

        return self.lm_head(self.ln_f(x))

    # ================= KV FAST GENERATION =================
    @torch.no_grad()
    def generate_kv(self, idx, max_new_tokens=200, temperature=0.7):

        self.eval()

        # reset cache
        self.k_cache = [None] * N_LAYER
        self.v_cache = [None] * N_LAYER

        out_tokens = []

        for step in range(max_new_tokens):

            x = self.token_emb(idx[:, -1:])  # only last token
            freqs = self.freqs[:idx.size(1)]

            h = x

            for i, block in enumerate(self.blocks):

                h = block.ln1(h)

                q, k, v = block.wqkv(h).chunk(3, dim=-1)

                q = q.view(1, -1, N_HEAD, N_EMBED // N_HEAD)
                k = k.view(1, -1, N_HEAD, N_EMBED // N_HEAD)
                v = v.view(1, -1, N_HEAD, N_EMBED // N_HEAD)

                q, k = apply_rope(q, freqs), apply_rope(k, freqs)

                # -------- KV CACHE --------
                if self.k_cache[i] is not None:
                    k = torch.cat([self.k_cache[i], k], dim=1)
                    v = torch.cat([self.v_cache[i], v], dim=1)

                self.k_cache[i] = k
                self.v_cache[i] = v

                attn = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=True
                )

                attn = attn.transpose(1, 2).contiguous().view(1, 1, N_EMBED)

                h = h + block.wo(attn)
                h = h + block.ffn(block.ln2(h))

            logits = self.lm_head(self.ln_f(h))
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            token_id = next_token.item()

            if token_id == 50256:
                break

            idx = torch.cat([idx, next_token], dim=1)
            out_tokens.append(token_id)

        return idx


# ---------------- LOAD ----------------
model = DeiaGPT().to(DEVICE)

if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print("✅ model loaded")
else:
    raise FileNotFoundError(MODEL_PATH)

# ---------------- BOT ----------------
bot = Bot(token=TOKEN)
dp = Dispatcher()

user_history = {}

@dp.message(Command("clear"))
async def clear(message: types.Message):
    user_history[message.from_user.id] = ""
    await message.answer("🧹 cleared")


@dp.message()
async def chat(message: types.Message):

    uid = message.from_user.id
    if uid not in ADMIN_IDS:
        return

    history = user_history.get(uid, "")

    prompt = f"{history}[USER] {message.text} [AGENT]"

    tokens = encode(prompt)
    tokens = tokens[-(BLOCK_SIZE - 50):]

    idx = torch.tensor([tokens], device=DEVICE)

    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):

        loop = asyncio.get_event_loop()

        out = await loop.run_in_executor(
            None,
            lambda: model.generate_kv(idx, 256)
        )

        reply_tokens = out[0, idx.size(1):].tolist()
        reply = decode(reply_tokens).strip()

        user_history[uid] = (prompt + reply)[-2000:]

        await message.answer(reply if reply else "...")


async def main():
    print("🚀 KV bot running")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())