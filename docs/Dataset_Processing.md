
# Dataset Processing & Provenance

This document should record dataset sources, transformations and provenance for any dataset used to train or fine-tune models in this repository.

Suggested sections:
- Sources: list original sources (URLs, data dumps) and license terms.
- Processing pipeline: scripts used (src/data_utils.py and Unsloth recipes), deduplication strategy, tokenization choices, and filtering thresholds.
- Splits: train/validation/test split procedure and sizes.
- Sampling: any sampling or up/down-weighting performed.
- Privacy & PII: steps taken to remove or redact personal data.
- Artifacts: file names and checksums (for reproducibility).

Example entry:
- Source: Common Crawl subset (link + date)
- Processing: run src/data_utils.process_jsonl_to_jsonl with config configs/dataset.yaml
- Output: data/processed/clean.jsonl (sha256: <sha256sum>)
