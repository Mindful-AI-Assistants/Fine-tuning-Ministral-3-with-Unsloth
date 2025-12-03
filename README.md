<br><br>





## Unsloth – Installation & Update (Portuguese Quick Guide)

<br><br>

This section provides a **clean, copy-ready Portuguese translation** of the core installation and update instructions from the official Unsloth documentation.


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
>
>
> * Fine-tuning & Reinforcement Learning for modern LLMs with **up to 2× faster training** and **70% less VRAM use**.
>




<br><br>




##  Table of Contents

* [ Get Started](#-get-started)
* [Fine-tuning Guide](#-fine-tuning-guide)
* [ Model Selection](#-model-selection)
* [ Tutorials](#-tutorials)
* [FAQ](#-faq)
* [Installation](#-installation)
* [ Dataset Guide](#-dataset-guide)
* [ Requirements](#-requirements)
* [ Inference & Deployment](#-inference--deployment)
* [ LoRA Hyperparameters](#-lora-hyperparameters)
* [⚡ Quickstart — CLI](#-quickstart--cli)
* [ Mistral 3 Quickstart](#-mistral-3-quickstart)
* [ Unsloth News](#-unsloth-news)
* [ Performance Benchmarks](#-performance-benchmarks)
* [Citation](#-citation)
* [License](#-license)




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
