from unittest.mock import Mock

import pytest
import torch
from transformers import (AutoTokenizer, AutoModelForMaskedLM, BatchEncoding,
                          BertForMaskedLM, BertTokenizerFast)

from app import get_top_k


@pytest.mark.parametrize("k", [5, 7])
def test_with_real_objects(k):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    sequence = "Hello [MASK]"
    res = get_top_k(sequence, tokenizer, model, k)

    assert isinstance(res, torch.Tensor)
    assert res.shape == (k,)


@pytest.mark.parametrize("k", [5, 7])
def test_with_mock_objects(k):
    sequence = "Hello [MASK]"
    vocab_size = 1000

    data = {"input_ids": torch.tensor([[101, 555, 103, 102]])}
    be = BatchEncoding(data=data)

    logits = torch.rand(1, 4, vocab_size)

    tokenizer_m = Mock(spec=BertTokenizerFast,
                       return_value=be,
                       mask_token_id=103)
    model_m = Mock(spec=BertForMaskedLM)
    model_m.return_value.logits = logits

    res = get_top_k(sequence,
                    tokenizer_m,
                    model_m,
                    k=k)

    assert isinstance(res, torch.Tensor)
    assert res.shape == (k,)
