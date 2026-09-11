from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_batch, TRAIN_TOKENS_PATH
from tokenizer.bpe import load_tokenizer, decode
from model.attention import MultiHeadAttention

TOKENIZER_DIRECTORY = Path("artifacts/tokenizer")

class TokenEmbedding(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_size
    ):
        super().__init__()

        # One row for every token in the vocabulary.
        # Every row contains embedding_size numbers.
        self.embedding_table = nn.Embedding(
            vocabulary_size,
            embedding_size
        )

    def forward(self, token_ids):

        # Converts every token ID into its embedding vector.
        token_embeddings = self.embedding_table(
            token_ids
        )

        return token_embeddings

if __name__ == "__main__":

    torch.manual_seed(42)

    # Two Shakespeare text windows.
    batch_size = 2

    # Sixteen BPE tokens in each window.
    context_length = 16

    # Each token will be represented using 32 numbers.
    embedding_size = 32

    # Four attention heads, each receiving 32 / 4 = 8 features.
    number_of_heads = 4

    # Load the tokenizer vocabulary.
    merges, vocabulary = load_tokenizer(
        output_directory=TOKENIZER_DIRECTORY
    )

    vocabulary_size = len(vocabulary)

    # Load all previously encoded Shakespeare token IDs.
    train_tokens = torch.load(
        TRAIN_TOKENS_PATH,
        map_location="cpu"
    )

   # Select two real Shakespeare context windows.
    input_token_ids, target_token_ids = get_batch(
        token_data=train_tokens,
        batch_size=batch_size,
        context_length=context_length
    )

    print("Vocabulary size:")
    print(vocabulary_size)

    print("\nInput token-ID shape:")
    print(input_token_ids.shape)

    print("\nFirst sequence token IDs:")
    print(input_token_ids[0])

    print("\nFirst sequence as BPE tokens:")

    first_sequence_tokens = [
        decode(
            token_ids=[token_id],
            vocabulary=vocabulary
        )
        for token_id in input_token_ids[0].tolist()
    ]
    print(first_sequence_tokens)

    print("\nFirst sequence reconstructed as text:")

    first_sequence_text = decode(
        token_ids=input_token_ids[0].tolist(),
        vocabulary=vocabulary
    )

    print(repr(first_sequence_text))

    # Create the trainable embedding table.
    token_embedding_layer = TokenEmbedding(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size
    )

    # Convert actual token IDs into embedding vectors.
    token_embeddings = token_embedding_layer(
        input_token_ids
    )

    print("\nEmbedding-table shape:")
    print(token_embedding_layer.embedding_table.weight.shape)

    print("\nToken-embedding tensor shape:")
    print(token_embeddings.shape)

    print("\nFirst token ID:")
    print(input_token_ids[0, 0])

    print("\nFirst token embedding:")
    print(token_embeddings[0, 0])

    # Pass the actual Shakespeare token embeddings into attention.
    attention = MultiHeadAttention(
        embedding_size=embedding_size,
        number_of_heads=number_of_heads,
        context_length=context_length
    )

    attention_output = attention(
        token_embeddings
    )

    print("\nAttention output shape:")
    print(attention_output.shape)
