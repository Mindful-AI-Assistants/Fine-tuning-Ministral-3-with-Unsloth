

"""
Tiny toy model used for CI smoke tests and local quick experiments.

This model is intentionally small and not optimized for real training.
It exists only to validate training/inference pipelines quickly on CPU.
"""

import torch
import torch.nn as nn


class MinimalLM(nn.Module):
    def __init__(self, vocab_size: int = 50400, d_model: int = 128, n_layers: int = 2, n_head: int = 4, d_ff: int = 512, max_seq: int = 1024):
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
