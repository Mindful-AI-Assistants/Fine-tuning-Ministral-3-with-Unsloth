"""
Toy training script to validate the training loop and environment quickly.
Uses synthetic data so it is safe to run in CI or on local machines.

Example:
python src/train_toy.py --epochs 1 --batch-size 8 --device cpu
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from model_toy import MinimalLM  # tiny model included in src/model_toy.py


class RandomDataset(Dataset):
    def __init__(self, vocab=50400, seq_len=64, n=512):
        self.vocab = vocab
        self.seq_len = seq_len
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab, (self.seq_len,))
        y = torch.roll(x, -1)
        return x, y


def train_epoch(model, loader, opt, scaler, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        with autocast(enabled=(device != "cpu")):
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    model = MinimalLM(vocab_size=50400, d_model=128, n_layers=2, n_head=4, d_ff=512).to(device)
    ds = RandomDataset(vocab=50400, seq_len=64, n=256)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    opt = AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=(device != "cpu"))

    os.makedirs("checkpoints/toy", exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, opt, scaler, device)
        print(f"Epoch {epoch} loss {loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/toy/model_epoch{epoch}.pt")


if __name__ == "__main__":
    main()
