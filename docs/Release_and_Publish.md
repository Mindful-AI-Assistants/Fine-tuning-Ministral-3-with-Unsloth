

# Release & Publish Guide

This document provides step-by-step instructions for preparing a release and publishing a model to Hugging Face Hub.

Checklist:
1. Verify training data provenance and license compatibility.
2. Prepare model card (MODEL_CARD.md) and link to DATASET_PROCESSING.md.
3. Convert and save weights using safetensors where possible.
4. Tag a release in git (e.g., v0.1.0) and add release notes.
5. Push the model to HF Hub:
   - Use `huggingface_hub` or `transformers` save_pretrained / push_to_hub.
6. Publish evaluation results and limitations.
7. Maintain a changelog and security contact.

Example: push a LoRA-adapter folder
- If using PEFT/LoRA, the saved folder contains only adapter weights:
  model.save_pretrained("outputs/finetune/epoch-2")
  from huggingface_hub import HfApi
  HfApi().upload_folder(repo_id="your-username/your-model", folder_path="outputs/finetune/epoch-2")
