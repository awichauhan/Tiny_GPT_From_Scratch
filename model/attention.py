import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttentionHead(nn.Module):
    def __init__(
            self,
            embedding_size,
            head_size,
            context_length
    ):
        super().__init__()

        self.query = nn.Linear(
            embedding_size,
            head_size,
            bias = False
        )

        self.key = nn.Linear(
            embedding_size,
            head_size,
            bias= False
        )

        self.value = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        causal_mask = torch.tril(
            torch.ones(context_length, context_length)
        )

        self.register_buffer(
            "causal_mask",
            causal_mask
        )

    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        attention_scores = (
            queries @ keys.transpose(-2,-1)
        )
        attention_scores = attention_scores * (  #scaled
            keys.shape[-1] ** -0.5
        )
        mask = self.causal_mask[
            :sequence_length,
            :sequence_length
        ]

        attention_scores = attention_scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        output = attention_weights @ values

        return output

class MultiHeadAttention(nn.Module):

    def __init__(
            self,
            embedding_size,
            number_of_heads,
            context_length
    ):
        super().__init__()

        if embedding_size % number_of_heads != 0:
            raise ValueError(
                "embedding_size must be divisible by number_of_heads"
            )
        head_size = embedding_size // number_of_heads

        self.heads = nn.ModuleList([
            CausalAttentionHead(
                embedding_size = embedding_size,
                head_size = head_size,
                context_length = context_length
            )
            for _ in range(number_of_heads)
        ])

        self.output_projection = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x):
        head_outputs = [
            head(x)
            for head in self.heads
        ]

        concatenated_output = torch.cat(
            head_outputs,
            dim=-1
        )

        output = self.output_projection(
            concatenated_output
        )

        return output

if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 2
    context_length = 5
    embedding_size = 8
    number_of_heads = 2

    x = torch.randn(
        batch_size,
        context_length,
        embedding_size
    )

    multi_head_attention = MultiHeadAttention(
        embedding_size=embedding_size,
        number_of_heads=number_of_heads,
        context_length=context_length
    )

    output = multi_head_attention(x)

    print("Input shape:")
    print(x.shape)

    print("\nMulti-head output shape:")
    print(output.shape)