>> > import torch
>> > from model import GPT
>> > from transformers import AutoModelForCausalLM
>> > hparams_names = [
    ...     "vocab_size",
    ...     "n_layer",
    ...     "n_embd",
    ...     "n_head",
    ...     "n_positions",
    ...     "attn_pdrop",
    ...     "embd_pdrop",
    ...     "resid_pdrop",
    ...     "layer_norm_epsilon",
    ...]
...
>> > model_name = "gpt2"
>> > model_official = AutoModelForCausalLM.from_pretrained(model_name, tie_word_embeddings=False)
>> > config_official = model_official.config
>> > config_official
>> > config_ours = {name: getattr(config_official, name) for name in hparams_names}
>> > config_ours
>> > model_ours = GPT(**config_ours)
>> > sum(p.numel() for p in model_ours.parameters())
>> > sum(p.numel() for p in model_official.parameters())
>> > _ = model_official.eval()
>> > _ = model_ours.eval()
>> > idx = torch.tensor([[1, 123, 52, 28]], dtype=torch.long)
>> > logits_official = model_official(idx).logits
>> > logits_ours = model_ours(idx)
>> > logits_official.shape
>> > logits_ours.shape
>> > torch.allclose(logits_ours, logits_official, rtol=0, atol=1e-3)
>> > (logits_ours - logits_official).abs().max()
>> > from utils import copy_model
>> > copy_model(model_official, model_ours)
>> > logits_official = model_official(idx).logits
>> > logits_ours = model_ours(idx)
>> > torch.allclose(logits_ours, logits_official, rtol=0, atol=1e-3)
>> > (logits_ours - logits_official).abs().max()
