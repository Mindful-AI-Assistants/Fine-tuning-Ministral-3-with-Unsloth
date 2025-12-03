
<br>


## Entendendo o LoRA (Low-Rank Adaptation)

<br><br>



> **Baseado na explicação do Professor [Samuel Fernando](https://www.linkedin.com/in/samuelfernando2030/)**
> *Senior AI/ML Engineer | Researcher | MSc Quantum Computing | LinkedIn Top Voice*

<br><br>



##  O que é LoRA?

LoRA (Low-Rank Adaptation) é uma técnica fundamental para **fine-tuning eficiente** de grandes modelos. Antes do hype da GenAI, pesquisadores como o **Professor [Samuel Fernando](https://www.linkedin.com/in/samuelfernando2030/)** já destacavam avanços essenciais como esse.

Publicado em 2021 — ainda na era GPT-3 — LoRA tornou possível adaptar modelos gigantes sem atualizar bilhões de parâmetros.

<br><br>


## 🧠 Como funciona?

Durante o fine-tuning tradicional, toda a matriz de pesos **W** é ajustada. O LoRA muda isso:

<br>

### 🥶 Congela a matriz base W


<br><br>


### ✔ Aprende apenas um ajuste ΔW de baixa dimensão:

```text
W' = W + ΔW
ΔW = B * A
```

Onde **A** e **B** são matrizes *low-rank*, com dimensões muito menores:

* A: (r × k)
* B: (d × r)
* com r ≪ d e k.

<br><br>


##  Exemplo numérico

Uma matriz **W** de 1000 × 1000 tem **1.000.000 parâmetros**.

Com LoRA, você atualiza só:

* A: r × 1000
* B: 1000 × r

Se r = 8 → **apenas 16.000 parâmetros**.
Uma redução enorme, preservando desempenho e economizando recursos.

<br><br>


##  Insight central

A mudança relevante na matriz W, após bilhões de passos de pré-treino, **vive em um subespaço de baixíssima dimensão**.
Essa é a genialidade do LoRA.

<br><br>


##  Impacto na GenAI

O ciclo inovação → implementação → produto encurtou drasticamente.
Em meses, LoRA virou biblioteca, padrão e base das técnicas modernas de PEFT.

<br><br>


## 🎓 Relevância acadêmica

Como destaca o Professor [**Samuel Fernando**](), a pesquisa acadêmica segue essencial: muito do que usamos hoje nasceu antes da popularização da GenAI.



