
# Minimal PyTorch scaffold and HF wrappers for Mistral-style models.
# This module provides safe, small building blocks to be adapted to real models.

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM, AutoConfig

class MinimalLM(nn.Module):
    """Small educational autoregressive transformer scaffold."""

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 2, n_head: int = 4, d_ff: int = 1024, max_seq: int = 1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_head, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor):
        b, seq = input_ids.shape
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def load_hf_model(model_id: str, device: str = "auto", load_in_8bit: bool = False):
    """Load a Hugging Face causal LM for fine-tuning or inference.

    Args:
      model_id: HF model identifier (e.g., 'mistralai/mistral-3-small').
      device: 'cpu', 'cuda' or 'auto'.
      load_in_8bit: if True, will attempt to use bitsandbytes 8-bit loading (requires bitsandbytes).
    """
    device_map = "auto" if device == "auto" else None
    kwargs = {"trust_remote_code": True}
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model
