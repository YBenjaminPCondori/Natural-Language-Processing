# Qwen Prompt Audit

- Qwen is prompt-based inference only; no Qwen fine-tuning is evidenced.
- Current parser code now accepts only exact allowed-label matches after whitespace/case normalisation.
- Retrieval few-shot prompts are configured to retrieve examples from the training split only.
- Current Qwen few-shot status from existing artifacts: `failed`.
- Rerun Qwen on CUDA to produce strict-parser zero/static/retrieval few-shot artifacts.
