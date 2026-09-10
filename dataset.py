from pathlib import Path

import torch


TRAIN_TOKENS_PATH = Path(
    "data/processed/train_tokens.pt"
)

VALIDATION_TOKENS_PATH = Path(
    "data/processed/validation_tokens.pt"
)


def create_context_window(token_data, context_length):
    """
    Create one input sequence and its shifted target sequence.
    """

    inputs = token_data[:context_length]

    targets = token_data[1:context_length + 1]

    return inputs, targets


def get_batch(token_data, batch_size, context_length):
    """
    Sample multiple random context windows from the token data.
    """

    maximum_start_position = (
        len(token_data) - context_length
    )

    start_positions = torch.randint(
        low=0,
        high=maximum_start_position,
        size=(batch_size,)
    )

    input_sequences = []
    target_sequences = []

    for start_position in start_positions:
        start_position = int(start_position)

        inputs = token_data[
            start_position:
            start_position + context_length
        ]

        targets = token_data[
            start_position + 1:
            start_position + context_length + 1
        ]

        input_sequences.append(inputs)
        target_sequences.append(targets)

    # Convert lists of 1D tensors into one 2D tensor
    input_batch = torch.stack(input_sequences)
    target_batch = torch.stack(target_sequences)

    return input_batch, target_batch


if __name__ == "__main__":

    torch.manual_seed(42)

    train_tokens = torch.load(
        TRAIN_TOKENS_PATH,
        map_location="cpu"
    )

    validation_tokens = torch.load(
        VALIDATION_TOKENS_PATH,
        map_location="cpu"
    )

    context_length = 8
    batch_size = 4

    # First examine one context window
    inputs, targets = create_context_window(
        token_data=train_tokens,
        context_length=context_length
    )

    print("One input context:")
    print(inputs)

    print("\nIts shifted targets:")
    print(targets)

    print("\nIndividual next-token examples:")

    for position in range(context_length):
        current_context = inputs[:position + 1]
        expected_next_token = targets[position]

        print(
            f"Input {current_context.tolist()} "
            f"should predict {expected_next_token.item()}"
        )

    # Now create a complete batch
    input_batch, target_batch = get_batch(
        token_data=train_tokens,
        batch_size=batch_size,
        context_length=context_length
    )

    print("\nInput batch:")
    print(input_batch)

    print("\nTarget batch:")
    print(target_batch)

    print("\nInput batch shape:")
    print(input_batch.shape)

    print("\nTarget batch shape:")
    print(target_batch.shape)

    # Shape verification
    assert input_batch.shape == (
        batch_size,
        context_length
    )

    assert target_batch.shape == (
        batch_size,
        context_length
    )

    assert input_batch.dtype == torch.long
    assert target_batch.dtype == torch.long

    print("\nContext-window and batch checks passed.")