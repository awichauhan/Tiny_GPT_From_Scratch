from pathlib import Path
from tokenizer.bpe import (
    train_bpe,
    build_vocabulary,
    save_tokenizer,
    load_tokenizer,
    encode,
    decode
)

TRAINING_DATA_PATH = Path("data/processed/train.txt")
TOKENIZER_DIRECTORY = Path("artifacts/tokenizer")

VOCABULARY_SIZE = 300

training_text = TRAINING_DATA_PATH.read_text(
    encoding="utf-8"
)

print("Training text characters: ")
print(len(training_text))

print("\nTraining BPE tokenizer...")
compressed_training_tokens, learned_merges = train_bpe(
    text=training_text,
    vocabulary_size= VOCABULARY_SIZE
)

vocabulary = build_vocabulary(learned_merges)

save_tokenizer(
    merges= learned_merges,
    vocabulary=vocabulary,
    output_directory=TOKENIZER_DIRECTORY
)

loaded_merges, loaded_vocabulary = load_tokenizer(
    output_directory=TOKENIZER_DIRECTORY
)

sample_text = training_text[:500]

encoded_sample = encode(
    text= sample_text,
    merges=loaded_merges
)

decoded_sample = decode(
    token_ids=encoded_sample,
    vocabulary=loaded_vocabulary
)

original_byte_count = len(training_text.encode("utf-8"))
compressed_token_count = len(compressed_training_tokens)

compression_ratio = compressed_token_count / original_byte_count

print("\nOriginal training bytes:")
print(original_byte_count)

print("\nCompressed training tokens:")
print(compressed_token_count)

print("\nCompression ratio:")
print(round(compression_ratio, 4))

print("\nNumber of learned merges:")
print(len(learned_merges))

print("\nActual vocabulary size:")
print(len(loaded_vocabulary))

print("\nOriginal sample:")
print(sample_text[:200])

print("\nDecoded sample:")
print(decoded_sample[:200])


# Verification
assert loaded_merges == learned_merges
assert decoded_sample == sample_text
assert len(loaded_vocabulary) == VOCABULARY_SIZE

print("\nReal-text tokenizer round trip passed.")