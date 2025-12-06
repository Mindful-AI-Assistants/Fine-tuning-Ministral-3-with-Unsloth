<br>
 
 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md

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


## [[Instalação]()

## 🧩 **Pip Install**

<br>

```bash
pip install unsloth
```

<br><br>
