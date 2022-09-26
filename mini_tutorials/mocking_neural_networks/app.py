import logging
import sys

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_top_k(sequence, tokenizer, model, k=10):
    """Get the top k most probable tokens to fill the gap with.

    Parameters
    ----------
    sequence : str
        String containing the [MASK] token.

    tokenizer : BertFastTokenizer
        Tokenizer.

    model : BertForMaskedLM
        Model.

    k : int
        Number of the top results to return.

    Returns
    -------
    top_vocab_indices : torch.Tensor
        1D tensor representing the indices of the top tokens.
    """
    batch_enc = tokenizer(sequence, return_tensors="pt")
    mask_ix = torch.where(batch_enc["input_ids"] == tokenizer.mask_token_id)[1]
    logits = model(**batch_enc).logits

    top_vocab_indices = torch.topk(logits[0, mask_ix.item(), :], k)[1]

    return top_vocab_indices


if __name__ == "__main__":
    logging.disable(logging.WARNING)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    sequence = sys.argv[1]

    top_indices = get_top_k(sequence, tokenizer, model, 5)
    top_tokens = [tokenizer.decode(torch.tensor([ix])) for ix in top_indices]

    winner = top_tokens[0]
    print(np.random.permutation(top_tokens))
    guess = input("Who do you think is the winner? ").strip()

    if guess == winner:
        print("You won!!!")
    else:
        print("You lost!!!")

    print("\nTrue ranking")
    for i, x in enumerate(top_tokens):
        print(i, x)
