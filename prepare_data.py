from pathlib import Path

raw_path = Path("data/raw/tiny_shakespeare.txt")
train_path = Path("data/processed/train.txt")
validation_path = Path("data/processed/validation.txt")

text = raw_path.read_text(encoding="utf-8")

split_index = int(len(text) * 0.90)

train_text = text[:split_index]
validation_text = text[split_index:]

train_path.write_text(train_text, encoding="utf-8")
validation_path.write_text(validation_text, encoding="utf-8")

print("Total characters: ", len(text))
print("Training characters: ", len(train_text))
print("Validation characters: ", len(validation_text))

print("\nTraining preview: ")
print(train_text[:300])

print("\nValidation preview: ")
print(validation_text[:300])