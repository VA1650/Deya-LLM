import torch
import tiktoken
import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.utils.chat_action import ChatActionSender
from model import GPT

# ================= КОНФИГУРАЦИЯ =================
TOKEN = 'ТВОЙ_ТОКЕН_БОТА'
ADMIN_IDS = [ТВОЙ_ID] 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Параметры должны совпадать с обучением
D_MODEL = 1024
HEADS = 16
LAYERS = 32
BLOCK_SIZE = 1024
VOCAB = 50257
MODEL_PATH = 'last.pt'

enc = tiktoken.get_encoding("gpt2")
def encode(t): return enc.encode(t, allowed_special={"<|endoftext|>"}, disallowed_special=())
def decode(t): return enc.decode(t)

# ================= ЗАГРУЗКА МОДЕЛИ =================

if os.path.exists(MODEL_PATH):
    print("Загрузка...")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Создаем модель через класс из model.py
    model = GPT(VOCAB, D_MODEL, HEADS, LAYERS, BLOCK_SIZE).to(DEVICE)
    
    # Загружаем веса (учитываем, что в last.pt лежит словарь с 'model', 'opt' и т.д.)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Дея v10.6 готова к общению!")
else:
    exit(f"❌ Ошибка: Файл {MODEL_PATH} не найден")

# ================= ЛОГИКА БОТА =================

bot = Bot(token=TOKEN)
dp = Dispatcher()
user_history = {}

@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    user_history[message.from_user.id] = ""
    await message.answer("Память очищена")

@dp.message()
async def handle_chat(message: types.Message):
    uid = message.from_user.id
    if uid not in ADMIN_IDS: return

    # Формируем историю. Если в обучении были [USER] и [AGENT], используем их.
    history = user_history.get(uid, "")
    prompt = f"{history}[USER] {message.text}\n[AGENT] "
    
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        # Кодируем и следим за размером окна контекста
        tokens_raw = encode(prompt)
        if len(tokens_raw) > BLOCK_SIZE - 200:
            tokens_raw = tokens_raw[-(BLOCK_SIZE - 200):]
            
        idx = torch.tensor([tokens_raw], device=DEVICE)
        
        # Генерация 
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            out = await loop.run_in_executor(None, lambda: model_generate(model, idx, 150))
        
        # Декодируем только то, что дописала модель
        new_tokens = out[0, idx.size(1):].tolist()
        full_reply = decode(new_tokens).strip()
        reply = full_reply.split("[USER]")[0].split("[AGENT]")[0].strip()
        
        # Обновляем историю для следующего шага
        user_history[uid] = f"{prompt}{reply}\n"[-1500:]
        
        if reply:
            await message.reply(reply)
        else:
            await message.answer("...")

def model_generate(model, idx, max_new_tokens, temperature=0.6):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # Top-K
        v, _ = torch.topk(logits, min(50, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        if idx_next.item() == 50256: # Конец текста
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот выключен.")
