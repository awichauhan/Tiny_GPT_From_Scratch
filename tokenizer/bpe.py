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

if __name__ == "__main__":
    sample_text = "abababab"

    token_ids = list(sample_text.encode("utf-8"))
    print("Original text: ")
    print(sample_text)

    print("\nInitial byte token IDs: ")
    print(token_ids)

    pair_counts = get_pair_counts(token_ids)

    print("\nPair Counts: ")
    print(pair_counts)

    most_frequent_pair = max(
        pair_counts,
        key= pair_counts.get
    )

    print("\nMost Frequent pairs: ")
    print(most_frequent_pair)

    new_token_id = 256
    merged_token_ids = merge_pair(
        token_ids=token_ids,
        pair_to_merge= most_frequent_pair,
        new_token_id= new_token_id
    )
    print("\nTokens after one BPE marge: ")
    print(merged_token_ids)

    print("\nOriginal token count: ")
    print(len(token_ids))

    print("\nToken count after merge: ")
    print(len(merged_token_ids))

"""
BPE basically first of all understand what are the repeated byte pairs in the text,, then replaces that repeated pair
with one extra token ID so the input stream gets shorter for model to learn on it"

multiple "ab", "ab" pairs having separate token IDs (1,2) for each pair, BPE will assign one unique and same token ID (256)
everytime this pair repeats. 
"""