<br>

 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]



<br><br>


# Unsloth Fine-Tuning Suite — Full Repository



<br><br>


> [!NOTE]
>
> * Complete fine-tuning environment for LLMs using Unsloth, including <br>
> * Ministral 3, Qwen, Llama, DeepSeek, Gemma, RL, Vision, GGUF export, and production deployment.
> <br>
> 

<br><br>


## [Includes]():

<br>

[-]() Jupyter notebooks

[-]()  Training, evaluation & inference scripts

[-]()  Dataset examples

🐳 Docker images

[-]() Full Unsloth support

🔥 Ministral 3 Quickstart



<br><br>



> [!NOTE]
>
> 
> * Source: [Unsloth – Install & Update](https://docs.unsloth.ai/get-started/install-and-update) 
>
> 



<br><br>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)(#license)
[![Python](https://img.shields.io/badge/Python-≤3.13-blue)](#installation)
[![Stars](https://img.shields.io/github/stars/unslothai/unsloth?style=social)](https://github.com/unslothai/unsloth)


<br><br>



> [!TIP]
>
>
> * Fine-tuning & Reinforcement Learning for modern LLMs with **up to 2× faster training** and **70% less VRAM use**.
>




<br><br>


Perfeito, Fabi — aqui está o **README completo**, ultra profissional, com o **TOC que você aprovou**, estilização GitHub Pro, ícones, seções completas e já integrado com **Ministral 3 + Unsloth + Fine-tuning + Notebooks + Scripts + Repo Structure**.

Pronto para **copiar/colar** em qualquer repositório.
(Todo em **inglês**, como pediu.)

---

# 🚀 Unsloth Fine-Tuning Suite — Full Repository

> **Complete fine-tuning environment** for LLMs using **Unsloth**, including
> **Ministral 3**, **Qwen**, **Llama**, **DeepSeek**, **Gemma**, RL, Vision, GGUF export, and production deployment.

Includes:

* 📘 Jupyter notebooks
* 🛠️ Training, evaluation & inference scripts
* 🧪 Dataset examples
* 🐳 Docker images
* 🧬 Full Unsloth support
* 🔥 Ministral 3 Quickstart

---

# 📚 Table of Contents

* [🚀 Introduction](#-introduction)
* [✨ Features](#-features)
* [📥 Installation](#-installation)

  * [Pip Install](#pip-install)
  * [Conda Install](#conda-install)
  * [Docker](#docker)
  * [Windows Support](#windows-support)
  * [Google Colab](#google-colab)
* [🧬 Fine-tuning Guide](#-fine-tuning-guide)

  * [What Model Should I Use?](#what-model-should-i-use)
  * [Dataset Structure](#dataset-structure)
  * [LoRA Hyperparameters](#lora-hyperparameters)
  * [Vision Fine-tuning](#vision-fine-tuning)
* [🔥 Ministral 3 Quickstart](#-ministral-3-quickstart)

  * [Available Models](#available-models)
  * [Training Notebook](#training-notebook)
  * [Inference Examples](#inference-examples)
* [📘 Notebooks](#-notebooks)

  * [Beginner Notebook](#beginner-notebook)
  * [Ministral 3 Notebook](#ministral-3-notebook)
  * [RL / Reasoning Notebooks](#rl--reasoning-notebooks)
* [🛠️ Scripts](#️-scripts)

  * [Training Script](#training-script)
  * [Evaluation Script](#evaluation-script)
  * [Inference Script](#inference-script)
* [📦 Repository Structure](#-repository-structure)
* [🖥️ Deployment](#-deployment)

  * [Ollama](#ollama)
  * [vLLM](#vllm)
  * [GGUF Export](#gguf-export)
* [⚠️ Troubleshooting](#️-troubleshooting)
* [💬 Community & Support](#-community--support)
* [📄 License](#-license)

---

# 🚀 Introduction

This repository provides a **complete environment** for fine-tuning modern LLMs using **Unsloth**, with support for:

* 🔥 Ministral 3 (all variants)
* 🦙 Llama 3 / 3.1 / 3.2 / 3.3
* 🐉 DeepSeek V3 / R1
* 🌠 Qwen 3 / 2.5 / VL / Coder
* ✨ Gemma 3
* 🧩 Phi models
* 🧠 Reinforcement Learning (DPO, ORPO, GRPO, KTO)

The repo includes:

* **Training notebooks**
* **Inference pipelines**
* **Dataset templates**
* **Docker images**
* **Export to GGUF / Ollama / vLLM**

---

## ✨ Features






-  Ultra-fast LoRA fine-tuning
-  
- FP16 / BF16 / FP8 support
- 
-  Ultra-long context (up to 500K tokens)
-  
 Vision fine-tuning (VLMs)

RL support (GRPO / DPO / ORPO / KTO)

 Export to **GGUF**, **Ollama**, **safetensors**

CPU, CUDA 11.8 / 12.1, AMD ROCm

---






# 📥 Installation

## Pip Install

```bash
pip install unsloth
```

## Conda Install

```bash
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y

conda activate unsloth_env
pip install unsloth
```

## Docker

```bash
docker pull unslothai/unsloth:latest
```

## Windows Support

Windows works via:

* WSL 2 (recommended)
* CUDA 12.1 GPUs
* CPU-only mode

## Google Colab

Free ready-to-use notebooks:
👉 [https://docs.unsloth.ai/get-started/beginner-start-here](https://docs.unsloth.ai/get-started/beginner-start-here)

---

# 🧬 Fine-tuning Guide

## What Model Should I Use?

* **Instruct models** → dialog, agents, chatbots
* **Base models** → reasoning, RAG, retrieval, embeddings
* Small datasets (<3K) → use **Instruct**
* Large datasets (>20K) → use **Base**

## Dataset Structure

Use the standard Unsloth chat template:

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

## LoRA Hyperparameters

Recommended:

```
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "v_proj"]
```

## Vision Fine-tuning

Supported for:

* Ministral 3 Vision
* Gemma Vision
* Qwen-VL

---

# 🔥 Ministral 3 Quickstart

## Available Models

* **Ministral 3 Small**
* **Ministral 3 Medium**
* **Ministral 3 14B** (fits on free Colab GPU)

## Training Notebook

👉 Included in repo under:

```
notebooks/ministral3_finetune.ipynb
```

## Example Training Code

```python
from unsloth import FastLanguageModel

model = FastLanguageModel.from_pretrained(
    "unsloth/ministral-3-14b",
    max_seq_length=4096,
)

model = FastLanguageModel.get_peft_model(model)
```

---

# 📘 Notebooks

### Beginner Notebook

```
notebooks/00_beginner_start_here.ipynb
```

### Ministral 3 Notebook

```
notebooks/ministral3_finetune.ipynb
```

### RL / Reasoning Notebooks

```
notebooks/rl/grpo_ministral3.ipynb
notebooks/rl/dpo_qwen3.ipynb
```

---

# 🛠️ Scripts

### Training Script

```
scripts/train.py
```

### Evaluation Script

```
scripts/eval.py
```

### Inference Script

```
scripts/infer.py
```

---

# 📦 Repository Structure

```
.
├── README.md
├── notebooks/
│   ├── 00_beginner_start_here.ipynb
│   ├── ministral3_finetune.ipynb
│   └── rl/
├── scripts/
│   ├── train.py
│   ├── infer.py
│   └── eval.py
├── data/
│   └── samples/
└── docker/
    └── Dockerfile
```

---

# 🖥️ Deployment

## Ollama

```bash
ollama create mymodel -f ollama_modelfile
```

## vLLM

```bash
python -m vllm.entrypoints.api_server --model ./output
```

## GGUF Export

```
unsloth convert --to-gguf output/
```

---

# ⚠️ Troubleshooting

Common issues:

* CUDA mismatch
* Out of memory → reduce LoRA rank
* Tokenizer mismatch → always use matching safetensors

---

# 💬 Community & Support

* Reddit: r/unsloth
* Docs: [https://docs.unsloth.ai](https://docs.unsloth.ai)
* HuggingFace Models: [https://huggingface.co/unsloth](https://huggingface.co/unsloth)

---

# 📄 License

MIT License (or change to your project license).

---





<br><br>


###  Instalação via Pip


<br>


```bash
pip install unsloth
```

<br><br>



### Atualizar para a última versão:


<br>

```bash
pip install --upgrade unsloth
```


<br><br>




## 🐍 Instalação com venv (Ambiente Virtual)


<br>


```bash
python3 -m venv unsloth_env
source unsloth_env/bin/activate     # Linux/macOS
unsloth_env\Scripts\activate        # Windows

pip install unsloth
```


<br><br>


### Google Colab (Recomendado)

<br>

```python
!pip install unsloth
```

<br>

### Depois, importe:

```python
from unsloth import FastModel, FastLanguageModel
```



<br><br>



## Instalação via Conda

```bash
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y

conda activate unsloth_env
pip install unsloth
```


<br><br>



## [Official Unsloth Notebooks]() (Tabela Completa)

Aqui está uma tabela pronta para README com **todos os notebooks oficiais do Unsloth**, incluindo fine-tuning, QLoRA, visão, RL, Ministal 3 e modelos específicos:

| Notebook                             | Descrição                                                | Link                                                                                                                                                                                                             |
| ------------------------------------ | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Beginner Start Here**              | Introdução, instalação, primeiros passos                 | [https://docs.unsloth.ai/get-started/beginner-start-here](https://docs.unsloth.ai/get-started/beginner-start-here)                                                                                               |
| **Fine-tuning Llama-3 (QLoRA)**      | Fine-tuning padrão com LoRA/QLoRA                        | [https://colab.research.google.com/github/unslothai/notebooks/blob/main/examples/fine_tune_llama3.ipynb](https://colab.research.google.com/github/unslothai/notebooks/blob/main/examples/fine_tune_llama3.ipynb) |
| **Ministral 3 Fine-tuning**          | Fine-tuning completo dos modelos Ministral 3 (com visão) | [https://docs.unsloth.ai/ministral-3-how-to-run-and-fine-tune](https://docs.unsloth.ai/ministral-3-how-to-run-and-fine-tune)                                                                                     |
| **Vision Fine-tuning**               | Fine-tuning de modelos de visão                          | [https://docs.unsloth.ai/vision-fine-tuning](https://docs.unsloth.ai/vision-fine-tuning)                                                                                                                         |
| **DeepSeek Fine-tuning**             | Treinar e rodar DeepSeek com Unsloth                     | [https://docs.unsloth.ai/deepseek-how-to-run-and-fine-tune](https://docs.unsloth.ai/deepseek-how-to-run-and-fine-tune)                                                                                           |
| **Gemma 3 Fine-tuning**              | Tutorial oficial para Gemma 3                            | [https://docs.unsloth.ai/gemma-3-how-to-run-and-fine-tune](https://docs.unsloth.ai/gemma-3-how-to-run-and-fine-tune)                                                                                             |
| **Qwen3 Fine-tuning**                | Treinar Qwen3 localmente com Unsloth                     | [https://docs.unsloth.ai/qwen3-how-to-run-and-fine-tune](https://docs.unsloth.ai/qwen3-how-to-run-and-fine-tune)                                                                                                 |
| **Qwen3-VL Vision**                  | Fine-tuning multimodal                                   | [https://docs.unsloth.ai/qwen3-vl-how-to-run-and-fine-tune](https://docs.unsloth.ai/qwen3-vl-how-to-run-and-fine-tune)                                                                                           |
| **gpt-oss Training**                 | Fine-tuning dos modelos gpt-oss                          | [https://docs.unsloth.ai/gpt-oss-how-to-run-and-fine-tune](https://docs.unsloth.ai/gpt-oss-how-to-run-and-fine-tune)                                                                                             |
| **Reinforcement Learning (GRPO)**    | Treinar modelos de raciocínio                            | [https://docs.unsloth.ai/Tutorial-train-your-own-reasoning-model-with-grpo](https://docs.unsloth.ai/Tutorial-train-your-own-reasoning-model-with-grpo)                                                           |
| **FP8 Reinforcement Learning**       | RL otimizado com FP8                                     | [https://docs.unsloth.ai/fp8-reinforcement-learning](https://docs.unsloth.ai/fp8-reinforcement-learning)                                                                                                         |
| **Ultra Long Context (500K tokens)** | Fine-tuning com contextos gigantes                       | [https://docs.unsloth.ai/500k-context-length-fine-tuning](https://docs.unsloth.ai/500k-context-length-fine-tuning)                                                                                               |



<br><br>



## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br>


#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />


<br><br>

<p align="center">  ────────────── ⊹🔭๋ ──────────────

<!--
<p align="center">  ────────────── 🛸๋*ੈ✩* 🔭*ੈ₊ ──────────────
-->

<br>

<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>
  

  
#
 
##### <p align="center">Copyright 2025 Mindful-AI-Assistants. Code released under the  [Apavhe Licencve.](https://github.com/Mindful-AI-Assistants/CDIA-Entrepreneurship-Soft-Skills-PUC-SP/blob/21961c2693169d461c6e05900e3d25e28a292297/LICENSE)
