# Minimal training runner demonstrating mixed precision and optional PEFT/LoRA integration.
import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from model import MinimalLM, load_hf_model

class RandomDataset(Dataset):
    def __init__(self, vocab=50257, seq_len=64, n=1024):
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
        with autocast():
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu","cuda","auto"])
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--use_hf", action="store_true", help="Load HF pretrained model (specify --hf_model)")
    parser.add_argument("--hf_model", type=str, default="", help="HF model id to load when --use_hf is set")
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device=="cuda" else "cpu"

    if args.use_hf and args.hf_model:
        model = load_hf_model(args.hf_model, device=device)
    else:
        model = MinimalLM(vocab_size=50257, d_model=256, n_layers=2, n_head=4, d_ff=1024)

    model = model.to(device)
    ds = RandomDataset(vocab=50257, seq_len=64, n=512)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, opt, scaler, device)
        print(f"Epoch {epoch} loss {loss:.4f}")
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_epoch{epoch}.pt"))

if __name__ == "__main__":
    main()
