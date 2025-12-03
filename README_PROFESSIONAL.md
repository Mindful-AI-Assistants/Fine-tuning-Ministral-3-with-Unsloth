
# Fine-tuning Mistral-3 with Unsloth — Professional Guide & Starter Scaffold

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](./LICENSE)
[![CI](https://github.com/Mindful-AI-Assistants/Fine-tuning-Ministral-3-with-Unsloth/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

This repository provides a production-oriented, reproducible scaffold to fine-tune Mistral-family causal language models using PyTorch and the Hugging Face ecosystem, with guidance on integrating Unsloth for dataset curation.

Highlights
- Reproducible data curation pipeline (Unsloth examples)
- Toy, safe CPU smoke tests (toy model + train)
- Production-ready fine-tuning example (Accelerate + PEFT/LoRA)
- CI workflow (lint + smoke tests), dev requirements and contributor guide
- Model card & release guidance for publishing to Hugging Face Hub

Quickstart (5–15 minutes)
1. Clone the repo:
   git clone https://github.com/Mindful-AI-Assistants/Fine-tuning-Ministral-3-with-Unsloth.git
   cd Fine-tuning-Ministral-3-with-Unsloth

2. Create virtualenv:
   python -m venv .venv
   source .venv/bin/activate

3. Install deps:
   pip install -r requirements.txt

4. Run smoke test (toy training):
   python src/train_toy.py --epochs 1 --batch-size 4 --device auto

5. Run demo inference:
   python src/inference.py --prompt "Write a short summary about AI in healthcare."

Repository layout (recommended)
- README_PROFESSIONAL.md (this file)
- README.md (original — preserved)
- CONTRIBUTING.md
- MODEL_CARD.md
- requirements.txt / requirements-dev.txt
- configs/
- src/
  - data_utils.py
  - model_toy.py
  - train_toy.py
  - train_hf.py
  - inference.py
- docs/
  - DATASET_PROCESSING.md
  - RELEASE_AND_PUBLISH.md
- .github/workflows/ci.yml
- tests/
- checkpoints/ (gitignored)

Core workflows (summarized)
- Data curation: use Unsloth recipes + src/data_utils.py to clean, dedupe and export JSONL/HF Datasets. Keep provenance in docs/DATASET_PROCESSING.md.
- Toy training: src/train_toy.py trains a tiny in-repo model on synthetic data for CI smoke tests.
- HF fine-tuning: src/train_hf.py demonstrates Accelerate+PEFT/LoRA flow with optional bitsandbytes 8-bit loading.
- Inference: src/inference.py provides a minimal generation example (greedy/sampling).

Safety & licensing
- Do not commit pretrained weights or raw/private datasets.
- Verify base model license (e.g., Mistral HF model) before redistribution.
- Include limitations and intended use in the model card when publishing.

CI & dev
- CI runs black/isort/flake8 and pytest smoke tests (fast, CPU-friendly).
- Use requirements-dev.txt for formatting and testing tools; enable pre-commit hooks for consistent formatting.

Publishing
- Create a model card (MODEL_CARD.md) with dataset provenance and limitations.
- Use safetensors to store weights where feasible.
- Tag releases and include changelog/benchmarks.

Next steps for maintainers
- Review README_PROFESSIONAL.md and decide whether to:
  - replace README.md, or
  - merge selected sections into README.md (recommended if existing README contains project-specific instructions).
- Confirm the Unsloth package name/version in requirements.txt (placeholder used).
- Optionally run the CI smoke tests after merge.

Useful links
- Transformers: https://huggingface.co/docs/transformers
- Accelerate: https://huggingface.co/docs/accelerate
- PEFT (LoRA): https://github.com/huggingface/peft
- bitsandbytes: https://github.com/TimDettmers/bitsandbytes
- PyTorch: https://pytorch.org

Contact
Open an issue with reproducible steps if you find problems or want further adaptations (examples, model-specific configs, translations to Portuguese, etc.).
