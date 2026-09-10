def get_pair_counts(token_ids):
    pair_counts = {}
    for pair in zip(token_ids, token_ids[1:]):
        pair_counts[pair] = pair_counts.get(pair, 0)+ 1

    return pair_counts

def merge_pair(token_ids, pair_to_merge, new_token_id):
    merged_token_ids = []
    index = 0
    while index < len(token_ids):
        if (
            index < len(token_ids) -1
            and token_ids[index] == pair_to_merge[0]
            and token_ids[index+1] == pair_to_merge[1]
        ):
            merged_token_ids.append(new_token_id)
            index+=2
        else:
            merged_token_ids.append(token_ids[index])
            index+=1
    return merged_token_ids

def train_bpe(text, vocabulary_size):
    if vocabulary_size < 256:
        raise ValueError(
            "Vocabulary size must be at least 256"
        )
    token_ids = list(text.encode("utf-8"))
    merges = {}

    for new_token_id in range(256, vocabulary_size):
        pair_counts = get_pair_counts(token_ids)
        if not pair_counts:
            break

        most_frequent_pair = max(
            pair_counts,
            key= pair_counts.get
        )
        frequency = pair_counts[most_frequent_pair]

        token_ids = merge_pair(
            token_ids= token_ids,
            pair_to_merge= most_frequent_pair,
            new_token_id=new_token_id
        )
        merges[most_frequent_pair] = new_token_id

        print(
            f"Merge {new_token_id}"
            f"{most_frequent_pair} -> {new_token_id}"
            f"| frequency: {frequency}"
            f"| tokens remaining: {len(token_ids)}"
        )
    return token_ids, merges

def build_vocabulary(merges):

    vocabulary = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    for pair, new_token_id in merges.items():  # .items() returns key-value pairs as tuples
        left_token_id, right_token_id = pair  # unpacks two tuple elements and store it in pair
        vocabulary[new_token_id] = (
            vocabulary[left_token_id]
            + vocabulary[right_token_id]
        )
    return vocabulary

def decode(token_ids, vocabulary):
    decoded_bytes = b"".join(
        vocabulary[token_ids]
        for token_id in token_ids
    )
    decoded_text = decoded_bytes.decode("utf-8")
    return decoded_text

if __name__ == "__main__":

    sample_text = "abababab"

    compressed_tokens, learned_merges = train_bpe(
        text=sample_text,
        vocabulary_size=258
    )

    print("\nOriginal text:")
    print(sample_text)

    print("\nOriginal byte tokens:")
    print(list(sample_text.encode("utf-8")))

    print("\nLearned merge rules:")
    print(learned_merges)

    print("\nFinal compressed tokens:")
    print(compressed_tokens)

"""
BPE basically first of all understand what are the repeated byte pairs in the text,, then replaces that repeated pair
with one extra token ID so the input stream gets shorter for model to learn on it"

multiple "ab", "ab" pairs having separate token IDs (1,2) for each pair, BPE will assign one unique and same token ID (256)
everytime this pair repeats. 
"""