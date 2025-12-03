"""
Minimal inference helper. Loads a HF model or local checkpoint and runs greedy/sampling generation.

Example:
python src/inference.py --model mistralai/mistral-3-small --prompt "Hello world" --device cpu
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate(model, tokenizer, prompt, max_new_tokens=128, do_sample=False, temperature=0.7, top_k=50, top_p=0.95, device="cpu"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="HF model id or local path")
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    print(generate(model, tokenizer, args.prompt, device=args.device))


if __name__ == "__main__":
    main()
