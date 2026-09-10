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


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size =2
    context_length = 4
    embedding_size = 8
    head_size = 4

    x = torch.randn(
        batch_size,
        context_length,
        embedding_size
    )

    attention_head = CausalAttentionHead(
        embedding_size=embedding_size,
        head_size= head_size,
        context_length=context_length
    )

    output = attention_head(x)

    print("Input shape: ")
    print(x.shape)

    print("\nCausal mask: ")
    print(attention_head.causal_mask)

    print("\nOutput shape: ")
    print(output.shape)