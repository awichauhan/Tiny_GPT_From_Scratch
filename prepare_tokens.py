from pathlib import Path
import torch
from tokenizer.bpe import(
    load_tokenizer,
    encode
)

TRAIN_TEXT_PATH = Path("data/processed/train.txt")
VALIDATION_TEXT_PATH = Path("data/processed/validation.txt")

TOKENIZER_DIRECTORY = Path("artifacts/tokenizer")

TRAIN_TOKENS_PATH = Path("data/processed/train_tokens.pt")
VALIDATION_TOKEN_PATH = Path("data/processed/validation_tokens.pt")

merges, vocabulary = load_tokenizer(
    output_directory=TOKENIZER_DIRECTORY
)

vocabulary_size = len(vocabulary)

train_text = TRAIN_TEXT_PATH.read_text(
    encoding="utf-8"
)

validation_text = VALIDATION_TEXT_PATH.read_text(
    encoding="utf-8"
)

train_token_ids = encode(
    text = train_text,
    merges = merges
)

validation_token_ids = encode(
    text = validation_text,
    merges = merges
)

train_tokens = torch.tensor(
    train_token_ids,
    dtype=torch.long
)

validation_tokens = torch.tensor(
    validation_token_ids,
    dtype=torch.long
)

assert train_tokens.ndim == 1
assert validation_tokens.ndim == 1

assert train_tokens.min() >= 0
assert validation_tokens.min() >= 0

assert train_tokens.max() < vocabulary_size
assert validation_tokens.max() < vocabulary_size

torch.save(
    train_tokens,
    TRAIN_TOKENS_PATH
)

torch.save(
    validation_tokens,
    VALIDATION_TOKEN_PATH
)

print("Vocabulary size:")
print(vocabulary_size)

print("\nTrain text characters:")
print(len(train_text))

print("\nTrain token count:")
print(len(train_tokens))

print("\nValidation text characters:")
print(len(validation_text))

print("\nValidation token count:")
print(len(validation_tokens))

print("\nTrain tensor shape:")
print(train_tokens.shape)

print("\nTrain tensor data type:")
print(train_tokens.dtype)

print("\nFirst 20 training token IDs:")
print(train_tokens[:20])

print("\nToken tensors saved successfully.")