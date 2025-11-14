# RProver

RProver is a lightweight, scriptable framework for generating mathematical proofs with LLMs and verifying them with configurable reviewers. It supports multiple proving strategies (naive, guided, hints, and ACE) and several verifier modes (standard, pessimistic, chunked pessimistic, and two‑phase judger). Runs save structured logs and sample outputs for inspection and follow‑up analysis.

**Key Features**
- Provers: `naive`, `gprover`, `hprover`, `aceprover`
- Verifiers: `standard`, `pessimistic`, `vpessimistic`, `pessimistic_judger`, `majority`
- Reviewer evaluation against a guider model (`--evaluate_reviewer`)
- Local datasets under `NP_dataset/` and HF datasets
- Async batch inference with token stats and reproducible logs

**Repository Structure**
- `main.py`: CLI entry point; provers, verifiers, orchestration, logging
- `utils/async_runner.py`: background asyncio loop helper
- `NP_dataset/`: local JSON/JSONL datasets for evaluation
- `reeval.py`: analyze disagreements between verifier and guider reviews
- `scripts/`: helper scripts (optional)
- `pyproject.toml`: Python 3.11 and dependencies; `uv.lock` for `uv` users
- Outputs: saved under `eval_logs/<UTC-Timestamp>/`

**Requirements**
- Python `>=3.11`
- An LLM API compatible with `litellm` (e.g., OpenAI)

**Setup**
- Create and activate a venv:
  - `python -m venv .venv && source .venv/bin/activate`
- Install dependencies:
  - `pip install -e .`
  - or with `uv`: `uv sync`

**Quick Start**
- Baseline (naive prover + standard reviewer):
  - `python main.py --eval_dataset NP_dataset/test_random.json --method naive --proof_model gpt-4o-mini --prover_base_url https://api.openai.com/v1 --prover_api_key $OPENAI_API_KEY`
- Guided methods require `--eval_model` and `--guider_model`:
  - `python main.py --eval_dataset NP_dataset/test_random.json --method gprover --proof_model gpt-4o-mini --eval_model gpt-4o-mini --guider_model gpt-4o-mini --prover_base_url https://api.openai.com/v1 --prover_api_key $OPENAI_API_KEY`

**CLI Overview**
- `--eval_dataset` (`-ed`): dataset path or name
- `--method`: `naive | gprover | hprover | aceprover`
- `--proof_model` (`-pm`): prover model name
- `--eval_model` (`-em`): reviewer/verifier model name
- `--guider_model` (`-gm`): guidance/ground‑truth model name
- `--reviewer`: `standard | pessimistic | vpessimistic | pessimistic_judger | majority`
- `--reviews`: parallel reviews for pessimistic/majority/judger (default 3)
- `--chunk_length`: lines per chunk for `vpessimistic` (default 7)
- `--evaluate_reviewer`: compare reviewer vs guider; writes metrics and samples
- `--reasoning_effort`: `minimal | low | medium | high` (passed to compatible models)
- `--log_dir`: base directory for logs (default `eval_logs`)
- API/base configuration (with fallbacks):
  - `--prover_base_url`, `--prover_api_key`
  - `--eval_base_url`, `--eval_api_key` (fall back to prover values)
  - `--guider_base_url`, `--guider_api_key` (fall back to eval then prover values)
- `--verifier_samples`: reuse problems/proofs and GT labels from an existing `verifier_samples.json`
- `--agenttrain`: enable agentic training mode (experimental)

**Datasets**
- Local JSON lists (each entry is a problem string):
  - `NP_dataset/train_full.json`, `train_3000.json`, `train_300.json`, `test_random.json`, `test_hard.json`
- Local JSONL files (expects `markdown_statement` per row):
  - `NP_dataset/qz_bench_train.jsonl`, `qz_bench_eval.jsonl`
- Hugging Face datasets:
  - `HuggingFaceH4/MATH-500` (uses `test` split; removes `solution`)
  - `AIME24/25` (combines AIME 2024 and 2025 sets)

**Outputs**
- Each run creates `eval_logs/<UTC-Timestamp>/` containing:
  - `logs.json`: CLI args, final `accuracy`, token stats (`average_*_tokens`), optional `verifier_evaluation`
  - `samples.json`: per‑sample problem, proof, evaluation label/text, and token counts
  - If `--reviewer progressive`: one file per refinement pass named `progressive_iteration_<n>_samples.json` capturing per‑iteration reviews, statuses, and chunk metadata
  - If `--evaluate_reviewer`:
    - `verifier_eval.json`: accuracy, precision, recall, F1 vs guider labels
    - `verifier_samples.json`: per‑sample comparison of reviewer vs guider
    - If the reviewer is `progressive`, `verifier_eval_progressive_iterations.json` adds accuracy/cost curves after each iteration (and the same data is embedded in `verifier_eval.json` under `progressive_iteration_metrics`)

**Reusing Proofs / Ground Truth**
- Use `--verifier_samples` to skip new proof generation and reuse prior problems/proofs and GT labels/texts:
  - `python main.py --verifier_samples eval_logs/1015T1203/verifier_samples.json --method naive --eval_model gpt-4o-mini --eval_base_url https://api.openai.com/v1 --eval_api_key $OPENAI_API_KEY`

**Disagreement Analysis (`reeval.py`)**
- Classify and summarize disagreements between reviewer and guider outputs stored in `verifier_samples.json`:
  - `python reeval.py --logdir eval_logs/<timestamp>`
- Writes:
  - `verifier_disagreements.json` and `verifier_disagreements.csv`
- Prints counts by class, actor attribution, and behavior tags.

**Security & Configuration**
- Do not commit API keys; pass via CLI flags or environment variables.
- Set API base/key per role; rely on documented fallbacks to reduce duplication.
- Respect `.gitignore` (e.g., `.venv/`, `eval_logs/`). Avoid embedding private datasets.

**Development & Testing**
- Recommended: `pytest` for unit tests under `tests/`.
- Suggested targets:
  - `prepare_dataset`
  - `LLMClient.infer_batch_async`
  - Prover/verifier orchestration
- Run: `pytest -q` (mock LLM calls; avoid real services).

**License**
- See `LICENSE`.
