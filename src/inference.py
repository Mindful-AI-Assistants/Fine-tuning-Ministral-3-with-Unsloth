

# Minimal inference demo using a HF tokenizer for tokenization and the MinimalLM scaffold.
import torch
from transformers import AutoTokenizer
from model import MinimalLM

def greedy_generate(model, tokenizer, prompt, max_new_tokens=32, device="cpu"):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        out = torch.cat([tokens, next_token], dim=1)
        return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MinimalLM(vocab_size=tokenizer.vocab_size, d_model=256, n_layers=2, n_head=4, d_ff=1024).to(device)
    print(greedy_generate(model, tokenizer, "Hello world"))
