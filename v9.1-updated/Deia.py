import os
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 4
ACCUMULATION_STEPS = 8

BLOCK_SIZE = 1024

MAX_ITERS = 50000

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1

DEVICE = "cuda"

N_EMBED = 768
N_HEAD = 12
N_LAYER = 32

DROPOUT = 0.1

INPUT_FILE = "input.txt"
TOKENIZER_PATH = "tokenizer.json"

SAVE_PATH = "deia_checkpoint.pth"
BEST_PATH = "deia_checkpoint_best.pth"

LOG_EVERY = 50
EVAL_EVERY = 500

VAL_RATIO = 0.01

GRAD_CLIP = 1.0


# ============================================================
# CUDA
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA не обнаружена. "
        "Эта версия train.py рассчитана на NVIDIA GPU."
    )


torch.set_float32_matmul_precision("high")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


print("=" * 60)
print("DEIA TRAINING")
print("=" * 60)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "CUDA:",
    torch.version.cuda
)

print()


# ============================================================
# TOKENIZER
# ============================================================

print("Загрузка tokenizer...")

if not os.path.exists(TOKENIZER_PATH):

    raise FileNotFoundError(
        f"Не найден {TOKENIZER_PATH}"
    )


tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


vocab_size = tokenizer.get_vocab_size()


print(
    f"Vocab size: {vocab_size}"
)


# ------------------------------------------------------------
# Special tokens
# ------------------------------------------------------------

def token_id(name):

    result = tokenizer.token_to_id(name)

    if result is None:

        print(
            f"WARNING: токен {name} не найден"
        )

    return result


PAD_ID = token_id("[PAD]")
UNK_ID = token_id("[UNK]")

BOS_ID = token_id("[BOS]")
EOS_ID = token_id("[EOS]")

USER_ID = token_id("[USER]")
AGENT_ID = token_id("[AGENT]")

EOT_ID = token_id("[EOT]")


print()
print("Special tokens:")

for name, value in [

    ("[PAD]", PAD_ID),
    ("[UNK]", UNK_ID),
    ("[BOS]", BOS_ID),
    ("[EOS]", EOS_ID),
    ("[USER]", USER_ID),
    ("[AGENT]", AGENT_ID),
    ("[EOT]", EOT_ID),

]:

    print(
        f"  {name:20} -> {value}"
    )


print()


# ============================================================
# ENCODE / DECODE
# ============================================================

def encode(text):

    return tokenizer.encode(
        text
    ).ids


def decode(tokens):

    return tokenizer.decode(
        tokens,
        skip_special_tokens=False
    )


# ============================================================
# MODEL
# ============================================================

def get_rope_freqs(
    dim,
    max_len,
    device
):

    inv_freq = 1.0 / (
        10000 ** (
            torch.arange(
                0,
                dim,
                2,
                device=device
            ).float()
            / dim
        )
    )


    t = torch.arange(
        max_len,
        device=device
    )


    freqs = torch.outer(
        t,
        inv_freq
    )


    return torch.polar(
        torch.ones_like(freqs),
        freqs
    )


def apply_rope(
    x,
    freqs
):

    x_complex = torch.view_as_complex(
        x.float().reshape(
            *x.shape[:-1],
            -1,
            2
        )
    )


    freqs = freqs[
        :x.size(1),
        None,
        :
    ]


    return torch.view_as_real(
        x_complex * freqs
    ).reshape(
        *x.shape
    ).type_as(x)


# ============================================================
# RMS NORM
# ============================================================

class RMSNorm(nn.Module):

    def __init__(
        self,
        dim
    ):

        super().__init__()

        self.scale = nn.Parameter(
            torch.ones(dim)
        )


    def forward(
        self,
        x
    ):

        return (

            x

            * torch.rsqrt(
                x.pow(2).mean(
                    -1,
                    keepdim=True
                )
                + 1e-6
            )

            * self.scale

        )


# ============================================================
# DEIA BLOCK
# ============================================================

class DeiaBlock(nn.Module):

    def __init__(
        self,
        n_embed,
        n_head
    ):

        super().__init__()


        self.n_head = n_head

        self.head_size = (
            n_embed // n_head
        )


        if n_embed % n_head != 0:

            raise ValueError(
                "N_EMBED должен делиться "
                "на N_HEAD"
            )


        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.wqkv = nn.Linear(
            n_embed,
            3 * n_embed,
            bias=False
        )


        self.wo = nn.Linear(
            n_embed,
            n_embed,
            bias=False
        )


        # ----------------------------------------------------
        # FFN
        # ----------------------------------------------------

        self.ffn = nn.Sequential(

            nn.Linear(
                n_embed,
                4 * n_embed,
                bias=False
            ),

            nn.GELU(),

            nn.Linear(
                4 * n_embed,
                n_embed,
                bias=False
            )

        )


        # ----------------------------------------------------
        # Norms
        # ----------------------------------------------------

        self.ln1 = RMSNorm(
            n_embed
        )

        self.ln2 = RMSNorm(
            n_embed
        )


    def forward(
        self,
        x,
        freqs
    ):

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        h = self.ln1(x)


        q, k, v = self.wqkv(
            h
        ).chunk(
            3,
            dim=-1
        )


        B, T, C = x.shape


        q = q.view(
            B,
            T,
            self.n_head,
            self.head_size
        )


        k = k.view(
            B,
            T,
            self.n_head,
            self.head_size
        )


        v = v.view(
            B,
            T,
            self.n_head,
            self.head_size
        )


        q = apply_rope(
            q,
            freqs
        )


        k = apply_rope(
            k,
            freqs
        )


        out = F.scaled_dot_product_attention(

            q.transpose(
                1,
                2
            ),

            k.transpose(
                1,
                2
            ),

            v.transpose(
                1,
                2
            ),

            is_causal=True,

            dropout_p=(
                DROPOUT
                if self.training
                else 0.0
            )

        )


        out = (

            out
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)

        )


        x = x + self.wo(
            out
        )


        # ----------------------------------------------------
        # FFN
        # ----------------------------------------------------

        x = x + self.ffn(
            self.ln2(x)
        )


        return x


# ============================================================
# DEIA GPT
# ============================================================

class DeiaGPT(nn.Module):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()


        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        self.token_emb = nn.Embedding(
            vocab_size,
            N_EMBED
        )


        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([

            DeiaBlock(
                N_EMBED,
                N_HEAD
            )

            for _ in range(N_LAYER)

        ])


        # ----------------------------------------------------
        # Final norm
        # ----------------------------------------------------

        self.ln_f = RMSNorm(
            N_EMBED
        )


        # ----------------------------------------------------
        # LM head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            N_EMBED,
            vocab_size,
            bias=False
        )


        # ----------------------------------------------------
        # Weight tying
        # ----------------------------------------------------

        self.lm_head.weight = (
            self.token_emb.weight
        )


        # ----------------------------------------------------
        # RoPE
        # ----------------------------------------------------

        self.register_buffer(

            "freqs",

            get_rope_freqs(

                N_EMBED // N_HEAD,

                BLOCK_SIZE,

                "cpu"

            )

        )


    def forward(
        self,
        idx,
        targets=None
    ):

        x = self.token_emb(
            idx
        )


        freqs = self.freqs[
            :idx.size(1)
        ]


        for block in self.blocks:

            x = gradient_checkpoint(

                block,

                x,

                freqs,

                use_reentrant=False

            )


        x = self.ln_f(
            x
        )


        logits = self.lm_head(
            x
        )


        loss = None


        if targets is not None:

            loss = F.cross_entropy(

                logits.reshape(
                    -1,
                    logits.size(-1)
                ),

                targets.reshape(
                    -1
                )

            )


        return logits, loss


    @torch.no_grad()
    def generate(

        self,

        idx,

        max_new_tokens,

        temperature=0.8,

        top_k=50,

        eos_token_id=None

    ):

        was_training = self.training

        self.eval()


        for _ in range(
            max_new_tokens
        ):


            idx_cond = idx[
                :,
                -BLOCK_SIZE:
            ]


            logits, _ = self(
                idx_cond
            )


            logits = logits[
                :,
                -1,
                :
            ]


            # ------------------------------------------------
            # Temperature
            # ------------------------------------------------

            if temperature <= 0:

                idx_next = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )

            else:

                logits = (
                    logits
                    / temperature
                )


                # --------------------------------------------
                # Top-K
                # --------------------------------------------

                if top_k is not None:

                    k = min(
                        top_k,
                        logits.size(-1)
                    )


                    values, _ = torch.topk(
                        logits,
                        k
                    )


                    min_value = values[
                        :,
                        -1:
                    ]


                    logits[
                        logits < min_value
                    ] = -float("inf")


                # --------------------------------------------
                # Sampling
                # --------------------------------------------

                probs = F.softmax(
                    logits,
                    dim=-1
                )


                idx_next = torch.multinomial(
                    probs,
                    num_samples=1
                )


            # ------------------------------------------------
            # Append
            # ------------------------------------------------

            idx = torch.cat(
                [
                    idx,
                    idx_next
                ],
                dim=1
            )


            # ------------------------------------------------
            # EOS
            # ------------------------------------------------

            if (

                eos_token_id is not None

                and idx_next.item()
                == eos_token_id

            ):

                break


        if was_training:

            self.train()


        return idx


# ============================================================
# DATA
# ============================================================

print("=" * 60)
print("Загрузка датасета")
print("=" * 60)


if not os.path.exists(
    INPUT_FILE
):

    raise FileNotFoundError(
        f"Не найден {INPUT_FILE}"
    )


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    text_data = f.read()


print(
    f"Символов: {len(text_data):,}"
)


print(
    "Токенизация..."
)


all_tokens = encode(
    text_data
)


print(
    f"Токенов: {len(all_tokens):,}"
)


if len(all_tokens) <= BLOCK_SIZE:

    raise RuntimeError(

        f"Датасет слишком маленький.\n"

        f"Токенов: {len(all_tokens)}\n"

        f"BLOCK_SIZE: {BLOCK_SIZE}"

    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

data = torch.tensor(
    all_tokens,
    dtype=torch.long
)


split = int(
    len(data)
    * (1.0 - VAL_RATIO)
)


train_data = data[
    :split
]


val_data = data[
    split:
]


print(
    f"Train tokens: "
    f"{len(train_data):,}"
)


print(
    f"Val tokens:   "
    f"{len(val_data):,}"
)


# ============================================================
# BATCH
# ============================================================

def get_batch(
    source
):

    max_start = (
        len(source)
        - BLOCK_SIZE
        - 1
    )


    ix = torch.randint(

        0,

        max_start,

        (
            BATCH_SIZE,
        )

    )


    x = torch.stack([

        source[
            i:
            i + BLOCK_SIZE
        ]

        for i in ix

    ])


    y = torch.stack([

        source[
            i + 1:
            i + BLOCK_SIZE + 1
        ]

        for i in ix

    ])


    return (

        x.to(
            DEVICE,
            non_blocking=True
        ),

        y.to(
            DEVICE,
            non_blocking=True
        )

    )


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 60)
print("Создание модели")
print("=" * 60)


model = DeiaGPT(
    vocab_size
).to(
    DEVICE
)


num_params = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"Параметров: "
    f"{num_params / 1e6:.1f}M"
)


print(
    f"Vocab:       {vocab_size}"
)

print(
    f"Embedding:   {N_EMBED}"
)

print(
    f"Heads:       {N_HEAD}"
)

print(
    f"Layers:      {N_LAYER}"
)

print(
    f"Block size:  {BLOCK_SIZE}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ============================================================
# CHECKPOINT
# ============================================================

start_iter = 0

best_loss = float(
    "inf"
)


def make_checkpoint(
    iteration,
    loss
):

    return {

        "state":
            model.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "iter":
            iteration,

        "loss":
            loss,

        "vocab_size":
            vocab_size,

        "block_size":
            BLOCK_SIZE,

        "n_embed":
            N_EMBED,

        "n_head":
            N_HEAD,

        "n_layer":
            N_LAYER,

        "tokenizer_path":
            TOKENIZER_PATH,

    }


if os.path.exists(
    BEST_PATH
):

    print()
    print(
        f"Найден checkpoint: "
        f"{BEST_PATH}"
    )


    ckpt = torch.load(

        BEST_PATH,

        map_location="cpu",

        weights_only=False

    )


    saved_vocab = ckpt.get(
        "vocab_size"
    )


    if (
        saved_vocab is not None
        and saved_vocab != vocab_size
    ):

        raise RuntimeError(

            "Несовместимый tokenizer!\n\n"

            f"Checkpoint vocab: "
            f"{saved_vocab}\n"

            f"Current vocab: "
            f"{vocab_size}\n\n"

            "Нужен tokenizer, с которым "
            "обучалась эта модель."

        )


    saved_block = ckpt.get(
        "block_size"
    )


    if (
        saved_block is not None
        and saved_block != BLOCK_SIZE
    ):

        raise RuntimeError(

            "Несовместимый BLOCK_SIZE!\n"

            f"Checkpoint: "
            f"{saved_block}\n"

            f"Current: "
            f"{BLOCK_SIZE}"

        )


    model.load_state_dict(
        ckpt["state"]
    )


    if "optimizer" in ckpt:

        optimizer.load_state_dict(
            ckpt["optimizer"]
        )


    start_iter = ckpt.get(
        "iter",
        0
    )


    best_loss = ckpt.get(
        "loss",
        float("inf")
    )


    print(
        f"Восстановлено!"
    )


    print(
        f"Шаг:  {start_iter}"
    )


    print(
        f"Loss: {best_loss:.4f}"
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()


    losses = []


    # Небольшое количество validation batches
    eval_iters = 20


    for _ in range(
        eval_iters
    ):

        xb, yb = get_batch(
            val_data
        )


        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16
        ):

            _, loss = model(
                xb,
                yb
            )


        losses.append(
            loss.item()
        )


    model.train()


    return sum(losses) / len(
        losses
    )


# ============================================================
# CHAT TEST
# ============================================================

def make_prompt(
    user_text
):

    return (
        "[USER] "
        + user_text
        + "\n[AGENT] "
    )


@torch.no_grad()
def test_generation():

    model.eval()


    prompt = make_prompt(
        "Привет! Как тебя зовут?"
    )


    tokens = encode(
        prompt
    )


    idx = torch.tensor(

        [tokens],

        dtype=torch.long,

        device=DEVICE

    )


    output = model.generate(

        idx,

        max_new_tokens=80,

        temperature=0.8,

        top_k=50,

        eos_token_id=EOT_ID

    )


    generated_tokens = output[
        0,
        len(tokens):
    ].tolist()


    answer = decode(
        generated_tokens
    )


    # --------------------------------------------------------
    # Не даём модели продолжить следующий USER
    # --------------------------------------------------------

    if "[USER]" in answer:

        answer = answer.split(
            "[USER]"
        )[0]


    print()
    print(
        "ДЕЯ:"
    )

    print(
        answer.strip()
    )


    model.train()


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

print(
    f"Start iteration: {start_iter}"
)

print(
    f"Max iterations:  {MAX_ITERS}"
)

print()


iter_start_time = time.time()


for iteration in range(

    start_iter,

    MAX_ITERS

):


    optimizer.zero_grad(
        set_to_none=True
    )


    loss_accum = 0.0


    # --------------------------------------------------------
    # Gradient accumulation
    # --------------------------------------------------------

    for _ in range(
        ACCUMULATION_STEPS
    ):


        xb, yb = get_batch(
            train_data
        )


        with torch.amp.autocast(

            "cuda",

            dtype=torch.bfloat16

        ):

            _, loss = model(
                xb,
                yb
            )


            loss = (
                loss
                / ACCUMULATION_STEPS
            )


        loss.backward()


        loss_accum += loss.item()


    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    grad_norm = torch.nn.utils.clip_grad_norm_(

        model.parameters(),

        GRAD_CLIP

    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer.step()


    current_loss = (
        loss_accum
        * ACCUMULATION_STEPS
    )


    # ========================================================
    # LOG
    # ========================================================

    if iteration % LOG_EVERY == 0:

        elapsed = (
            time.time()
            - iter_start_time
        )


        print()

        print(
            f"--- Iteration "
            f"{iteration}"
        )


        print(
            f"Train loss: "
            f"{current_loss:.4f}"
        )


        print(
            f"Grad norm: "
            f"{grad_norm:.4f}"
        )


        print(
            f"Time: "
            f"{elapsed:.1f}s"
        )


        print(
            "VRAM allocated:",
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )


        print(
            "VRAM reserved:",
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )


        # ----------------------------------------------------
        # Generation
        # ----------------------------------------------------

        test_generation()


        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        gc.collect()

        torch.cuda.empty_cache()


    # ========================================================
    # VALIDATION
    # ========================================================

    if (

        iteration != 0

        and iteration % EVAL_EVERY == 0

    ):


        print()
        print(
            "Validation..."
        )


        val_loss = estimate_loss()


        print(
            f"Validation loss: "
            f"{val_loss:.4f}"
        )


        # ----------------------------------------------------
        # Save regular checkpoint
        # ----------------------------------------------------

        ckpt = make_checkpoint(
            iteration,
            val_loss
        )


        torch.save(
            ckpt,
            SAVE_PATH
        )


        print(
            f"Checkpoint сохранён: "
            f"{SAVE_PATH}"
        )


        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_loss:

            best_loss = val_loss


            ckpt = make_checkpoint(

                iteration,

                val_loss

            )


            torch.save(

                ckpt,

                BEST_PATH

            )


            print(
                "✨ Новый лучший checkpoint!"
            )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 60)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_loss:.4f}"
)


final_ckpt = make_checkpoint(

    MAX_ITERS,

    best_loss

)


torch.save(
    final_ckpt,
    SAVE_PATH
)


print(
    f"Сохранено: {SAVE_PATH}"
)
