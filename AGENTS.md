# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: CLI entry and core provers/verifier.
- `utils/async_runner.py`: background asyncio loop helper.
- `NP_dataset/`: local JSON datasets for evaluation.
- `scripts/`: helper scripts (optional).
- `pyproject.toml`: Python 3.11 and dependencies; `uv.lock` for `uv` users.
- Outputs: logs and samples saved under `eval_logs/<UTC-Timestamp>/` when running.

## Build, Test, and Development Commands
- Setup (venv): `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -e .`  (or `uv sync` if using `uv`).
- Run (example):
  - `python main.py --eval_dataset NP_dataset/test_random.json --method naive --proof_model gpt-4o-mini --prover_base_url https://api.openai.com/v1 --prover_api_key $OPENAI_API_KEY`
  - For guided methods, also set `--eval_model` and `--guider_model`.
- Logs: inspect `eval_logs/<timestamp>/logs.json` and `samples.json`.
  - Choose verifier: `--reviewer standard` (default) or `--reviewer pessimistic`.
  - Pessimistic rounds: `--reviews 3` (default 3 parallel checks).
  - Evaluate reviewer vs guider: add `--evaluate_reviewer` and set `--guider_model`.
    - Outputs: `verifier_eval.json` (metrics) and `verifier_samples.json` (per-sample comparison) in the same log dir.
  - API/base fallbacks:
    - `--eval_base_url`/`--eval_api_key` fall back to prover flags.
    - `--guider_base_url`/`--guider_api_key` fall back to eval (then prover) flags.

## Coding Style & Naming Conventions
- Python 3.11, PEP 8, 4-space indentation, type hints where practical.
- Use `snake_case` for functions/vars, `CamelCase` for classes.
- Prefer `pathlib.Path`, structured `logging`, and small, cohesive functions.
- Keep public CLI args documented in `main.py`.

## Testing Guidelines
- Framework: `pytest` is recommended (add under `tests/`).
- Naming: `tests/test_*.py`; unit tests for `prepare_dataset`, `LLMClient.infer_batch_async`, and provers’ orchestration.
- Run: `pytest -q` (ensure API calls are mocked; avoid hitting real services).

## Commit & Pull Request Guidelines
- Commits: clear, imperative subject; scope-specific; reference issues.
- PRs: concise description, rationale, commands used, and sample output paths (e.g., `eval_logs/1015T1203/`).
- Include screenshots/log snippets when changing runtime behavior.

## Security & Configuration Tips
- Do not commit API keys or secrets; pass via CLI or env vars.
- Respect `.gitignore` (e.g., `.venv/`, `eval_logs/`). Avoid embedding private datasets.

## Agent-Specific Instructions
- Scope applies repository-wide. Keep patches minimal and focused.
- Follow style and structure above; avoid unrelated refactors; update docs when changing CLI or outputs.
