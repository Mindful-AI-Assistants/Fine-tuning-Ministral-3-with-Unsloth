<br>
 
 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]

<br><br>


# <p align="center"> Fine-tuning Ministral-3 with [Unsloth]() 🔥 Guia Completo
### <p align="center"> Treino acelerado, otimizado e barato usando Unsloth + Ministral-3.

<br><br>



Este repositório fornece um ambiente completo, rápido e moderno para fine-tuning, inferência, curadoria de dados e exportação de modelos Ministral-3, Llama, Qwen, Gemma, DeepSeek e variantes, utilizando o ecossistema Unsloth.

Inclui notebooks prontos, scripts de treino, exemplos de datasets, Docker, exportação para GGUF/Ollama/vLLM e suporte a Reinforcement Learning (GRPO, DPO, ORPO, KTO).



<br><br><br>



### <p align="center"> [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](#license) [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-Mindful%20AI%20%20Assistants-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants) [![Python](https://img.shields.io/badge/Python-≤3.13-blue)](#installation)


<br><br><br>



> [!NOTE]
>
> [-]()  Ambiente completo de fine-tuning para LLMs usando Unsloth, incluindo <br>
> [-]()  Ministral 3, Qwen, Llama, DeepSeek, Gemma, RL, Vision, exportação GGUF e deployment em produção. <br>
> [-]()  Fonte: [Unsloth – Instalação & Atualização](https://docs.unsloth.ai/get-started/install-and-update)  <br> <br>

<br><br>


## [Inclui]():

<br>

[-]() Jupyter notebooks

[-]() Scripts de treinamento, avaliação e inferência

[-]() Exemplos de datasets

[-]() 🐳 Imagens Docker

[-]() Suporte completo ao Unsloth

[-]() 🔥 Quickstart do Ministral 3




> [!TIP]
>
> * **Fine-tuning & Reinforcement Learning** para LLMs modernos com **até 2× mais velocidade de treino** e **70% menos uso de VRAM**. <br>
> * **Ambiente completo de fine-tuning** para LLMs usando **Unsloth**, incluindo <br>
> * **Ministral 3**, **Qwen**, **Llama**, **DeepSeek**, **Gemma**, RL, Vision, exportação GGUF e deployment em produção. <br>
>  <br>
>  

<br><br><br>



## Indíce

- [Introdução](#introdução)
- [Features](#features)
- [Instalação](#instalação)
  - [Pip](#pip)
  - [Conda](#conda)
  - [Docker](#docker)
  - [Windows](#windows)
  - [Google Colab](#google-colab)
- [Guia de Fine-tuning](#guia-de-fine-tuning)
  - [Escolha de Modelo](#escolha-de-modelo)
  - [Estrutura de Dataset](#estrutura-de-dataset)
  - [Hiperparâmetros LoRA](#hiperparâmetros-lora)
  - [Vision Fine-tuning](#vision-fine-tuning)
- [Ministral-3 Quickstart](#ministral-3-quickstart)
- [Notebooks](#notebooks)
- [Scripts](#scripts)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Deployment & Export](#deployment--export)
  - [Ollama](#ollama)
  - [vLLM](#vllm)
  - [GGUF](#gguf)
- [Troubleshooting](#troubleshooting)
- [Comunidade & Suporte](#comunidade--suporte)
- [License](#license)


<br><br>


## [Introdução]()


Este repositório consolida um ambiente robusto e padronizado para:

* **Fine-tuning eficiente com LoRA/QLoRA**
* **Aprendizado por Reforço (GRPO, DPO, ORPO, KTO)**
* **Treinamento em Visão (VLMs)**
* **Exportação para GGUF e deployment em CPU/GPU**
* **Inferência otimizada com Unsloth e vLLM**
* **Ambiente de desenvolvimento reprodutível (Docker + Conda)**


<br><br>

## [Suporte completo para:]()

* **Ministral-3 (todos os tamanhos)**
* Llama 3.x
* Qwen 2.5 / 3 / VL
* Gemma 3
* DeepSeek V3 / R1
* Phi 3
* Vision LLMs


<br><br>

##  [Features]()


* ⚡ *Até 2× mais rápido* que frameworks tradicionais
*  *70% menos VRAM* com QLoRA
*  Suporte completo para **Fine-tuning em Visão**
*  Suporte para **RL (GRPO / DPO / ORPO / KTO)**
*  *Contexto ultra-longo* (até 500K tokens)
*  Exportação para **GGUF**, **Ollama**, **vLLM**, **safetensors**
*  Docker + scripts padronizados
*  Notebooks para Colab / uso local
*  CPU, CUDA 11.8 / 12.1, AMD ROCm



<br><br>


## [Instalação]()

###  [Pip Install]()

<br>

```bash
pip install unsloth
```

<br><br>

### 🐍 [Conda Install]()

<br>

```bash
conda create --name unsloth_env python=3.11 -y
conda activate unsloth_env
pip install unsloth
```

<br><br>

### 🐳 [Docker]()

<br>

```bash
docker pull unslothai/unsloth:latest
```

<br><br>

### [Suporte Windows]()

<br>

✔ Via WSL2 (recomendado)

✔ CUDA 12.1

✔ Apenas CPU


<br><br>


##  [Google Colab]()

<br>

Notebooks oficiais:
https://docs.unsloth.ai/get-started/beginner-start-here


<br>

[Instalação rápida:]()

<br>

```bash
!pip install unsloth
```

<br><br>

##  [Guia de Fine-tuning]()


###  [Qual modelo escolher ?]()

<br>

| [Tarefa]()                | Modelo recomendado |
| --------------------- | ------------------ |
| [Chat / Agentes]()        | Instruct           |
| [Raciocínio]()            | Base               |
| [Dataset pequeno (<3k)]() | Instruct           |
| [Dataset grande (>20k)]() | Base               |



<br><br>


## [Estrutura do Dataset]()

### [Formato padrão (JSONL):]()

<br>

```json
{
  "messages": [
    {"role": "user", "content": "Olá"},
    {"role": "assistant", "content": "Oi! Como posso ajudar?"}
  ]
}
```

<br><br>

## [Hiperparâmetros LoRA]()

### [Recomendação inicial:]()

<br>

```ìni
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
```

<br><br>


## [Vision Fine-tuning]()

### [Suporte para:]()

<br>

* Ministral-3 Vision

* Qwen-VL

* Gemma Vision


<br><br>


## 🔥 [Ministral-3 Quickstart]()

###  [Exemplos de modelos suportados]()

<br>

* Ministral-3 Small

* Ministral-3 Medium

* Ministral-3 14B (cabe no Colab Free com QLoRA)


<br><br>

## [Notebook oficial do repo]()


```bash
notebooks/ministral3_finetune.ipynb
```

<br><br>

## Código de treino (exemplo)()

<br>


```python
from unsloth import FastLanguageModel

model = FastLanguageModel.from_pretrained(
    "unsloth/ministral-3-14b",
    max_seq_length=4096,
)

model = FastLanguageModel.get_peft_model(model)
```


<br><br>

## [Notebooks]()

<br>

| [Notebook]()                | [Descrição]()                     | [Link]()                                   |
| ----------------------- | ----------------------------- | -------------------------------------- |
| [Beginner Start Here]()     | Introdução e primeiros passos | notebooks/00_beginner_start_here.ipynb |
| [Ministral-3 Fine-tuning]() | Treino completo               | notebooks/ministral3_finetune.ipynb    |
| [GRPO RL]()                 | Raciocínio com RL             | notebooks/rl/grpo_ministral3.ipynb     |
| [DPO Qwen]()                | RL DPO                        | notebooks/rl/dpo_qwen3.ipynb           |


<br><br>


## [Oficial Unsloth Notebooks]() 

<br>

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

## [Scripts]()

<br>

```bash
scripts/train.py
scripts/eval.py
scripts/infer.py
```


<br><br>


## [Estrutura do Repositório]()

<br>


```javascript
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


<br><br>

## [Deployment & Export()

<br>

```bash
ollama create mymodel -f ollama_modelfile
```

<br><br>


## [vLLM]()


<br>

```bash
python -m vllm.entrypoints.api_server --model ./output
```


<br><br>


## [Exportar para GGUF]()

<br>

```bash
unsloth convert --to-gguf output/
```

<br><br>


## ❗ [Resolução de Problemas()

<br>

* Incompatibilidade de CUDA

* OOM → reduza o rank do LoRA

* Incompatibilidade do Tokenizer → use matching safetensors


<br><br>


## [Comunidade & Suporte]()

<br>

* [Reddit](r/unsloth)

* [Docs oficiais](https://docs.unsloth.ai)

* [Modelos - Hugging Face](https://huggingface.co/unsloth)

* [Discord oficial]()


<br><br>

## [Licença]()

<br>

Apache 2.0



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
 


<br><br>

##### <p align="center"> Code released under the  [Apache Licencve.](https://github.com/Mindful-AI-Assistants/CDIA-Entrepreneurship-Soft-Skills-PUC-SP/blob/21961c2693169d461c6e05900e3d25e28a292297/LICENSE)

