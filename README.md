<br>

#  Unsloth — Fast Fine-tuning & Reinforcement Learning for LLMs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](#license)
[![Python](https://img.shields.io/badge/Python-≤3.13-blue)](#installation)
[![Stars](https://img.shields.io/github/stars/unslothai/unsloth?style=social)](https://github.com/unslothai/unsloth)

Fine-tuning & Reinforcement Learning for modern LLMs with **up to 2× faster training** and **70% less VRAM use**.

<br><br>

<br><br># 📚 Table of Contents

* [🚀 Get Started](#-get-started)
* [🧬 Fine-tuning Guide](#-fine-tuning-guide)
* [❓ Model Selection](#-model-selection)
* [📘 Tutorials](#-tutorials)
* [🤔 FAQ](#-faq)
* [📥 Installation](#-installation)
* [📈 Dataset Guide](#-dataset-guide)
* [🛠 Requirements](#-requirements)
* [🖥 Inference & Deployment](#-inference--deployment)
* [🧠 LoRA Hyperparameters](#-lora-hyperparameters)
* [⚡ Quickstart — CLI](#-quickstart--cli)
* [🌟 Mistral 3 Quickstart](#-mistral-3-quickstart)
* [🦥 Unsloth News](#-unsloth-news)
* [🥇 Performance Benchmarks](#-performance-benchmarks)
* [📜 Citation](#-citation)
* [📄 License](#-license)


<br><br>


<br><br># 🚀 Get Started

**Beginner? Start here!**
Perguntas mais comuns antes do seu primeiro fine-tune.

👉 Pergunte também na comunidade: r/unsloth (Reddit)


#


## Fine-tuning Guide

Aprenda como treinar modelos passo a passo.
Inclui: SFT, QLoRA, FP8 training e GRPO.

<br><br>

## Model Selection

* Instruct vs Base
* Tamanho ideal do dataset
* Quando usar RAG vs Fine-tuning

<br><br>


## 📘 Tutorials

* Fine-tuning DeepSeek
* Parametrização para Gemma 3
* Como rodar modelos localmente, via Ollama, GGUF, SGLang, vLLM

---

# 🤔 FAQ

* Quando fine-tunar?
* Diferença entre SFT, DPO, GRPO
* Como evitar OOM (out-of-memory)?

---

# 📥 Installation

## Linux / WSL

```bash
pip install unsloth
```

## Windows

*Requer PyTorch previamente instalado.*

Guia completo: Windows Guide.

## Docker

```bash
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
```

---

# 📈 Dataset Guide

* Como organizar dataset SFT
* DPO vs SFT formats
* Captura de dados
* Boas práticas

---

# 🛠 Requirements

Compatível com:

* NVIDIA GPUs (2018+)
* AMD
* Intel
* CUDA Capability ≥ 7.0

---

# 🖥 Inference & Deployment

* Export GGUF
* Roda via llama.cpp
* Roda via vLLM, SGLang, Ollama
* Salvamento de checkpoints

---

# 🧠 LoRA Hyperparameters

Comportamento dos parâmetros:
r, alpha, target_modules, dropout, RSLORA, LoftQ, etc.

---

# ⚡ Quickstart — CLI

Exemplo de fine-tuning **gpt-oss-20b**:

```python
from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

max_seq_length = 2048
dataset = load_dataset("json", data_files={"train": ".../unified_chip2.jsonl"}, split="train")

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gpt-oss-20b",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=60,
        logging_steps=1,
    ),
)

trainer.train()
```

---

# 🌟 Mistral 3 Quickstart

> **Nova seção solicitada — estilo idêntico ao README oficial**

Treine **Mistral 3** (7B/8B/22B/large) usando QLoRA ou full-finetuning.

## ▶️ Instalação

```bash
pip install unsloth
```

## ▶️ Carregar modelo Mistral 3

```python
from unsloth import FastModel, FastLanguageModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/mistral-3-8b",
    max_seq_length=4096,
    load_in_4bit=True,
)
```

## ▶️ Aplicar LoRA otimizado

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
```

## ▶️ Treinar

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="mistral3-output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=120,
        logging_steps=1,
    ),
)

trainer.train()
```

## ▶️ Exportar para GGUF

```python
model.save_pretrained("mistral3-gguf")
```

---

# 🦥 Unsloth News

* FP8 Reinforcement Learning
* DeepSeek OCR Fine-tuning
* Novo Docker super otimizado
* Suporte completo para TTS, Vision, GRPO, GSPO, DPO, ORPO…

---

# 🥇 Performance Benchmarks

Comparação Unsloth vs HuggingFace (FA2):

* **2× mais rápido**
* **Até 75% menos VRAM**
* **Longest context: 340k tokens** (para GPUs 80GB)

---

# 📜 Citation

```bibtex
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {http://github.com/unslothai/unsloth},
  year = {2023}
}
```

---

# 📄 License

Apache-2.0

---

Se quiser, Fabi, posso também:

✨ transformar isso em **README.md pronto para copiar/colar**
✨ criar versão com **ícones coloridos + estilização GitHub Pro**
✨ criar estrutura de repositório inteira (README + notebooks + scripts)

É só pedir.
