"""
Fine-tuning script using Hugging Face Transformers + Accelerate + PEFT (LoRA).
This is a well-documented example — adapt hyperparameters and dataset loading to your needs.
"""
import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, get_scheduler
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True, help="path to JSONL or HF dataset id")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset: support local JSONL or HF dataset id
    if Path(args.dataset_path).exists():
        dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=args.max_seq_length)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Model load
    model_kwargs = {"trust_remote_code": True}
    if args.load_in_8bit:
        model_kwargs.update({"load_in_8bit": True, "device_map": "auto"})
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if args.load_in_8bit:
        # prepare model if using 8-bit training
        model = prepare_model_for_kbit_training(model)

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    # Accelerator + Dataloaders
    train_dataloader = torch.utils.data.DataLoader(tokenized, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    total_steps = len(train_dataloader) * args.num_train_epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if step % 50 == 0:
                print(f"Epoch {epoch} step {step} loss {loss.item():.4f}")

        # Save checkpoint (PEFT saves adapter weights small)
        output_dir = Path(args.output_dir) / f"epoch-{epoch}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.use_lora:
            model.save_pretrained(output_dir)
        else:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
