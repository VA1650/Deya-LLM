from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tokenizer = Tokenizer(
    models.BPE(
        unk_token="[UNK]"
    )
)

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
        "[AGENT]",
        "[USER]",
        "[EOT]",
    ]
)

tokenizer.train(
    ["input.txt"],
    trainer
)

tokenizer.save("tokenizer.json")

print(
    "Токенизатор обучен."
)

print(
    "Vocab size:",
    tokenizer.get_vocab_size()
)

for token in [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[AGENT]",
    "[USER]",
    "[EOT]",

]:
    print(
        token,
        "=>",
        tokenizer.token_to_id(token)
    )
