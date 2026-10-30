import torch
import tiktoken
import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.chat_action import ChatActionSender

from model import GPT

# ================= CONFIG =================
TOKEN = ""  # Укажите ваш токен бота
ADMIN_IDS = []  # Укажите ID администраторов

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL = 1024
HEADS = 16
LAYERS = 32
BLOCK_SIZE = 1024
VOCAB = 50257
MODEL_PATH = "last.pt"

# ================= TOKENIZER =================
enc = tiktoken.get_encoding("gpt2")

SPECIAL_TOKENS = {"[USER]", "[AGENT]", "<|endoftext|>"}

def encode(text):
    return enc.encode(text, allowed_special=SPECIAL_TOKENS)

def decode(tokens):
    return enc.decode(tokens)

# ================= LOAD MODEL =================
print("🤖 loading model...")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

model = GPT(VOCAB, D_MODEL, HEADS, LAYERS, BLOCK_SIZE).to(DEVICE)

state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(state, strict=False)

model.eval()
print("✅ model ready")

# ================= BOT =================
bot = Bot(token=TOKEN)
dp = Dispatcher()

# Храним историю в виде токенов, а не сырого текста!
user_history = {}

# ================= SAFETY TOKENS =================
# Только реальный токен конца текста (никаких разбиений строк)
BAD_TOKENS = {enc.eot_token}

# ================= GENERATION =================
def generate_stream(model, idx, max_new_tokens=200, temperature=0.6):
    tokens = []

    for i in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Top-K filtering
        top_k = 100
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        token_id = idx_next.item()

        # STOP CONDITIONS
        if token_id in BAD_TOKENS:
            break

        idx = torch.cat((idx, idx_next), dim=1)
        tokens.append(token_id)

        # stream step
        if i % 4 == 0:
            yield tokens.copy(), idx

    yield tokens, idx


# ================= HANDLER =================
@dp.message(Command("clear"))
async def clear(message: types.Message):
    user_history.pop(message.from_user.id, None)
    await message.answer("🧹 История очищена")


@dp.message()
async def chat(message: types.Message):
    uid = message.from_user.id
    if uid not in ADMIN_IDS:
        return

    # Достаем прошлую историю токенов или пустой список
    history_tokens = user_history.get(uid, [])

    # Формируем новое сообщение в токенах
    new_prompt_part = f"[USER] {message.text}\n[AGENT] "
    new_tokens = encode(new_prompt_part)

    # Объединяем историю и новый запрос
    tokens = history_tokens + new_tokens

    # Безопасно обрезаем по токенам с запасом под ответ
    max_context_len = BLOCK_SIZE - 200
    if len(tokens) > max_context_len:
        tokens = tokens[-max_context_len:]

    idx = torch.tensor([tokens], device=DEVICE)

    # Отправляем сообщение для стриминга
    msg = await message.answer("💭 thinking...")

    loop = asyncio.get_event_loop()

    def run():
        return list(generate_stream(model, idx, 200))

    results = await loop.run_in_executor(None, run)

    last_text = ""

    for token_buf, current_idx in results:
        try:
            text = decode(token_buf)
        except:
            continue

        # cleanup (страховка от мусора в выводе)
        text = text.replace("[USER]", "").replace("[AGENT]", "").strip()

        if text == last_text:
            continue

        last_text = text

        try:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=msg.message_id,
                text=text if text else "..."
            )
        except:
            pass

    reply = last_text.strip()

    # Если модель ничего не вернула, обновим текст заглушки на многоточие
    if not reply:
        try:
            await bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=msg.message_id,
                text="..."
            )
        except:
            pass

    # Сохраняем в историю всю цепочку в виде токенов (включая ответ модели)
    reply_tokens = encode(reply + "<|endoftext|>\n")
    user_history[uid] = (tokens + reply_tokens)[-1500:]  # Ограничение по токенам истории

# ================= RUN =================
async def main():
    print("🚀 KV bot running")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
