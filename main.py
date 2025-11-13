import argparse
import json
import asyncio
import re
from pathlib import Path
from utils.async_runner import AsyncLoopThread
from datasets import load_dataset, Dataset, concatenate_datasets
from litellm import acompletion
import logging
from datetime import datetime, timezone
import random
from tqdm import tqdm

ASYNC_LOOP = AsyncLoopThread()

def extract_xml_content(text: str, tag: str):
    flags = re.DOTALL | 0
    pattern = rf"<{re.escape(tag)}(?:\s+[^>]*)?\s*>(.*?)</\s*{re.escape(tag)}\s*>"

    last_content = None
    for m in re.finditer(pattern, text, flags):
        last_content = m.group(1)

    if last_content is None:
        return None
    return last_content.strip()

def find_boxed(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a

def strip_think_simple(s: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)

def get_current_log_path(log_dir: str):
    ts = datetime.now(timezone.utc).strftime("%m%dT%H%M")
    logdir = Path(log_dir) / ts
    return logdir

def _load_jsonl_problems(jsonl_path: Path, content_keys: tuple[str, ...] = ("markdown_statement",)) -> list[str]:
    """Load problems from a JSONL file.

    Each line must be a JSON object. The text of the problem is extracted
    from the first available key in `content_keys`.
    """
    logger = logging.getLogger("dataset")
    problems: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s line %d", jsonl_path, i)
                continue
            content = None
            for key in content_keys:
                if key in obj and isinstance(obj[key], str):
                    content = obj[key]
                    break
            if content is None:
                logger.warning("No problem content key %s found at %s line %d", content_keys, jsonl_path, i)
                continue
            problems.append(content)
    logger.info("Loaded %d problems from %s", len(problems), jsonl_path)
    return problems

# For dataset decryption purposes!
def _derive_keystream(canary: str, length: int) -> bytes:
    import hashlib
    from math import ceil
    digest = hashlib.sha256(canary.encode("utf-8")).digest()
    if length <= len(digest):
        return digest[:length]
    repeats = ceil(length / len(digest))
    return (digest * repeats)[:length]

def _xor_bytes(data: bytes, key_stream: bytes) -> bytes:
    if len(data) != len(key_stream):
        raise ValueError("Data and keystream must be the same length for XOR.")
    return bytes([a ^ b for a, b in zip(data, key_stream)])

def _deserialize_field(text: str):
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        return text


def decrypt_str(input_str, canary):
    import base64
    if input_str == "":
        return ""
    ct = base64.b64decode(input_str)
    ks = _derive_keystream(canary, len(ct))
    pt = _xor_bytes(ct, ks)
    text = pt.decode("utf-8")
    return _deserialize_field(text)

def decrypt_h2eval_sample(example):
    if "canary" not in example:
        raise ValueError("Missing canary field `canary`.")
    canary = example["canary"]
    if not isinstance(canary, str):
        raise ValueError("Canary should be a string.")

    target_fields = ["question", "model_response_by_step", "human_labels", "human_labels_first_error_idx"]
    for k, v in example.items():
        if k in target_fields and isinstance(v, str):
            try:
                example[k] = decrypt_str(v, canary)
            except Exception:
                example[k] = v

    return example

def prepare_dataset(dataset_path):
    """
    this function prepares datasets according to the given path
    """
    logger = logging.getLogger("dataset")
    logger.info("preparing dataset at path: %s", dataset_path)
    if dataset_path == "NP_dataset/train_full.json" or dataset_path == "NP_dataset/train_3000.json" or dataset_path == "NP_dataset/test_hard.json" or dataset_path == "NP_dataset/test_random.json" or dataset_path == "NP_dataset/train_300.json":
        with Path(dataset_path).open("r", encoding="utf-8") as f:
            problems = json.load(f)
        ds = Dataset.from_dict({"problem": problems})
    elif dataset_path in {
        "NP_dataset/qz_bench_train.jsonl",
        "NP_dataset/qz_bench_eval.jsonl",
    }:
        problems = _load_jsonl_problems(Path(dataset_path), content_keys=("markdown_statement",))
        ds = Dataset.from_dict({"problem": problems})
    elif dataset_path == "NP_dataset/gradingbench.csv":
        ds = load_dataset("csv", data_files=dataset_path)
        ds = ds["train"].select(range(300))
        ds = ds.rename_column("Problem", "problem")
        ds = ds.rename_column("Response", "proof")
        gt_evals = [int(e["Points"]) > 6 for e in ds]
        ds = ds.add_column("gt_eval", gt_evals)
    elif dataset_path == "HuggingFaceH4/MATH-500":
        ds = load_dataset(dataset_path)
        ds = ds.remove_columns(["solution"])
        ds = ds["test"]
    elif dataset_path == "AIME24/25":
        ds24 = load_dataset("Maxwell-Jia/AIME_2024")
        ds24 = ds24["train"]
        ds24 = Dataset.from_dict(
            {
                "problem": [e["Problem"] for e in ds24],
                "answer": [str(e["Answer"]) for e in ds24]
            }
        )
        ds25_1 = load_dataset("opencompass/AIME2025", "AIME2025-I")
        ds25_2 = load_dataset("opencompass/AIME2025", "AIME2025-II")
        ds25 = concatenate_datasets([ds25_1['test'], ds25_2['test']])
        ds25 = Dataset.from_dict(
            {
                "problem": [e["question"] for e in ds25],
                "answer": [e["answer"] for e in ds25]
            }
        )
        ds = concatenate_datasets([ds24, ds25])
    elif dataset_path == "Salesforce/Hard2Verify":
        ds = load_dataset(dataset_path, split="test")
        ds = ds.map(decrypt_h2eval_sample)
        ds = ds.rename_column("question", "problem")
        proofs = ["\n".join(e["model_response_by_step"]) for e in ds]
        ds = ds.add_column("proof", proofs)
        gt_evals = [e["human_labels_first_error_idx"] < 0 for e in ds]
        ds = ds.add_column("gt_eval", gt_evals)
        # ds = ds.rename_column("human_labels_first_error_idx", "error_idx")
    elif dataset_path == "INSAIT-Institute/OPC":
        # We only use test set of this dataset
        ds = load_dataset(dataset_path, split="test")
        ds = ds.rename_column("solution", "proof")
        gt_evals = [e["score"][0] for e in ds]
        ds = ds.add_column("gt_eval", gt_evals)
    else:
        raise NotImplementedError(f"Unknown dataset name or path: {dataset_path}")

    logger.info("completed preparing dataset at: %s", dataset_path)

    return ds

class LLMClient():
    def __init__(self, api_base, api_key, model):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.input_tokens = []
        self.comp_tokens = []
        self.last_input_tokens = []
        self.last_comp_tokens = []

    def _supports_enable_thinking(self) -> bool:
        model_name = (self.model or "").lower()
        return "deepseek" in model_name or "qwen" in model_name

    async def _infer_one(self,
                         messages,
                         sem: asyncio.Semaphore,
                         **kwargs):
        backoff = 1.0
        while True:
            try:
                async with sem:
                    resp = await acompletion(
                        model="openai/"+self.model,
                        messages=messages,
                        api_base=self.api_base,
                        api_key=self.api_key,
                        drop_params=True,
                        temperature=1.0,
                        timeout=3600,
                        num_retries=7,
                        **kwargs)
                return resp
            except Exception as e:
                msg = str(e).lower()
                if any(k in msg for k in ["rate", "timeout", "overloaded", "temporarily"]):
                    await asyncio.sleep(backoff + random.random() * 0.2)
                    backoff = min(backoff * 2, 60)
                    continue
                # raise
                return None

    async def infer_batch_async(self,
                                all_messages,
                                concurrency: int = 8,
                                show_progress: bool = True,
                                **kwargs) -> list[str]:
        logger = logging.getLogger("evaluator")
        logger.info("running batch inference on %d samples", len(all_messages))
        sem = asyncio.Semaphore(concurrency)
        ALLOWED_PARAM_KEYS = {"reasoning_effort", "thinking"}
        infer_params = {k: v for k, v in kwargs.items() if k in ALLOWED_PARAM_KEYS}
        enable_thinking = kwargs.get("enable_thinking")
        if enable_thinking is not None and self._supports_enable_thinking():
            infer_params["enable_thinking"] = enable_thinking
        async def _run_one(index: int, messages):
            try:
                r = await self._infer_one(messages, sem, **infer_params)
            except Exception as e:
                r = e
            return index, r

        tasks = [asyncio.create_task(_run_one(i, messages)) for i, messages in enumerate(all_messages)]
        raw_results = [None] * len(all_messages)

        pbar = tqdm(total=len(all_messages), desc="LLM batch", leave=False) if show_progress else None
        try:
            for t in asyncio.as_completed(tasks):
                idx, r = await t
                raw_results[idx] = r
                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()

        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                raise RuntimeError(f"Task {i} failed") from r
        logger.info("completed batch inference on %d samples",  len(all_messages))
        completions = [r.choices[0].message["content"] if r is not None else "" for r  in raw_results]
        batch_input_tokens = [r.usage.prompt_tokens for r in raw_results if r is not None]
        batch_comp_tokens = [r.usage.completion_tokens for r in raw_results if r is not None]
        self.last_input_tokens = batch_input_tokens
        self.last_comp_tokens = batch_comp_tokens
        self.input_tokens.extend(batch_input_tokens)
        self.comp_tokens.extend(batch_comp_tokens)
        return completions

class Verifier():
    def __init__(self, api_base, api_key, model):
        self.client = LLMClient(api_base, api_key, model)

    def __call__(self, problems, completions, **kwargs):
        all_messages = [
            [
                {"role": "system", "content": (
                    "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
                )},
                {"role": "user", "content": (
                    "Here is a math problem and a candidate solution of it, and you need to verify the correctness of this solution. Please check each of the following:\n\n"
                    "1. The provided content is indeed a math problem and its corresponding solution, rather than unrelated material supplied by mistake.\n"
                    "2. The solution actually derives the conclusion required by the original problem.\n"
                    "3. Every step of calculation and formula derivation in the solution is correct.\n"
                    "4. The hypotheses (conditions) and conclusions of any theorems used are correctly matched and applied.\n"
                    "5. The solution relies only on the conditions given in the problem and does not introduce any additional assumptions to obtain the conclusion.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the solution as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness (e.g., invalid step, wrong theorem usage without required conditions, uncorrected calculation error leading to a wrong result), treat the solution as incorrect.\n\n"
                    "Response requirements: If the solution is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any."
                    " If the solution is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error."
                    " Do not include any restatement of the entire solution or problem.\n\n"
                    f"<problem>{p}</problem>\n\n"
                    f"<answer>{strip_think_simple(c if isinstance(c, str) else c[0]['content'])}</answer>"
                )}
            ]
            for (p, c) in zip(problems, completions)
        ]
        results = ASYNC_LOOP.run(self.client.infer_batch_async(all_messages, **kwargs))
        rewards = [1.0 if extract_xml_content(r, "verification") == "true" else 0.0 for r in results]
        return rewards, results

class PessimisticJudger():
    """
    Runs multiple parallel reviews using the same checklist as Verifier,
    then asks a judger to double-check negative findings and produce a final verdict.
    """
    def __init__(self, api_base, api_key, model, review_times: int = 3):
        self.client = LLMClient(api_base, api_key, model)
        self.review_times = max(1, review_times)

    def _review_messages(self, problems, completions):
        messages = []
        for (p, c) in zip(problems, completions):
            answer = strip_think_simple(c if isinstance(c, str) else c[0]['content'])
            base = [
                {"role": "system", "content": (
                    "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
                )},
                {"role": "user", "content": (
                    "Here is a math problem and a candidate solution of it, and you need to verify the correctness of this solution. Please check each of the following:\n\n"
                    "1. The provided content is indeed a math problem and its corresponding solution, rather than unrelated material supplied by mistake.\n"
                    "2. The solution actually derives the conclusion required by the original problem.\n"
                    "3. Every step of calculation and formula derivation in the solution is correct.\n"
                    "4. The hypotheses (conditions) and conclusions of any theorems used are correctly matched and applied.\n"
                    "5. The solution relies only on the conditions given in the problem and does not introduce any additional assumptions to obtain the conclusion.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the solution as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness (e.g., invalid step, wrong theorem usage without required conditions, uncorrected calculation error leading to a wrong result), treat the solution as incorrect.\n\n"
                    "Response requirements: If the solution is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any."
                    " If the solution is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error."
                    " Do not include any restatement of the entire solution or problem.\n\n"
                    f"<problem>{p}</problem>\n\n"
                    f"<answer>{answer}</answer>"
                )}
            ]
            for _ in range(self.review_times):
                messages.append(base)
        return messages

    def _build_judge_messages(self, problems, completions, grouped_reviews):
        judge_messages = []
        for (p, c), reviews in zip(zip(problems, completions), grouped_reviews):
            answer = strip_think_simple(c if isinstance(c, str) else c[0]['content'])
            negatives = [strip_think_simple(r) for r in reviews if extract_xml_content(r, "verification") == "false"]
            neg_block = "\n".join(f"- Review {i+1}: {r}" for i, r in enumerate(negatives)) if negatives else "(none)"
            judge_messages.append([
                {"role": "system", "content": (
                    "You are a rigorous mathematics proof judger. You will receive a problem, a candidate solution, and several negative peer reviews that claim there are errors."
                    " Sequentially double-check each alleged error: decide whether it actually harms the correctness of the proof. If any harmful error is confirmed, the proof is incorrect."
                )},
                {"role": "user", "content": (
                    "Task: Examine the negative reviews and determine whether the candidate solution remains correct."
                    " Apply the following policy: minor, easily fixable issues do NOT invalidate correctness; critical errors that undermine the argument DO."
                    " Provide a brief justification (2-3 sentences). If the solution is correct overall (possibly with minor issues), append `<verification>true</verification>` and briefly note minor issues if any; otherwise append `<verification>false</verification>`.\n\n"
                    f"<problem>{p}</problem>\n\n"
                    f"<answer>{answer}</answer>\n\n"
                    f"<negative_reviews>\n{neg_block}\n</negative_reviews>"
                )}
            ])
        return judge_messages

    def __call__(self, problems, completions, **kwargs):
        # Phase 1: parallel reviews
        review_messages = self._review_messages(problems, completions)
        all_reviews = ASYNC_LOOP.run(self.client.infer_batch_async(review_messages, **kwargs))
        # Group reviews per problem
        k = self.review_times
        grouped = [all_reviews[i * k:(i + 1) * k] for i in range(len(problems))]

        # Phase 2: judgement
        judge_messages = self._build_judge_messages(problems, completions, grouped)
        judge_results = ASYNC_LOOP.run(self.client.infer_batch_async(judge_messages, **kwargs))

        rewards = [1.0 if extract_xml_content(r, "verification") == "true" else 0.0 for r in judge_results]
        return rewards, judge_results

class PessimisticVerifier():
    """
    Runs multiple parallel reviews using the same checklist as Verifier.
    Instead of asking a judger, it treats the FIRST review that reports an error
    (`<verification>false</verification>`) as the final verdict for that proof.
    """
    def __init__(self, api_base, api_key, model, review_times: int = 3):
        self.client = LLMClient(api_base, api_key, model)
        self.review_times = max(1, review_times)

    def _review_messages(self, problems, completions):
        messages = []
        for (p, c) in zip(problems, completions):
            answer = strip_think_simple(c if isinstance(c, str) else c[0]['content'])
            base = [
                {"role": "system", "content": (
                    "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
                )},
                {"role": "user", "content": (
                    "Here is a math problem and a candidate solution of it, and you need to verify the correctness of this solution. Please check each of the following:\n\n"
                    "1. The provided content is indeed a math problem and its corresponding solution, rather than unrelated material supplied by mistake.\n"
                    "2. The solution actually derives the conclusion required by the original problem.\n"
                    "3. Every step of calculation and formula derivation in the solution is correct.\n"
                    "4. The hypotheses (conditions) and conclusions of any theorems used are correctly matched and applied.\n"
                    "5. The solution relies only on the conditions given in the problem and does not introduce any additional assumptions to obtain the conclusion.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the solution as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness (e.g., invalid step, wrong theorem usage without required conditions, uncorrected calculation error leading to a wrong result), treat the solution as incorrect.\n\n"
                    "Response requirements: If the solution is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any."
                    " If the solution is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error."
                    " Do not include any restatement of the entire solution or problem.\n\n"
                    f"<problem>{p}</problem>\n\n"
                    f"<answer>{answer}</answer>"
                )}
            ]
            for _ in range(self.review_times):
                messages.append(base)
        return messages

    def __call__(self, problems, completions, **kwargs):
        # Only perform parallel reviews and take the first error as verdict
        review_messages = self._review_messages(problems, completions)
        all_reviews = ASYNC_LOOP.run(self.client.infer_batch_async(review_messages, **kwargs))
        k = self.review_times
        grouped = [all_reviews[i * k:(i + 1) * k] for i in range(len(problems))]

        final_reviews = []
        rewards = []
        for reviews in grouped:
            # find first negative review
            first_negative = None
            for r in reviews:
                if extract_xml_content(r, "verification") == "false":
                    first_negative = r
                    break
            if first_negative is not None:
                rewards.append(0.0)
                final_reviews.append(first_negative)
            else:
                rewards.append(1.0)
                # fallback: take the first positive review's content
                final_reviews.append(reviews[0] if reviews else "")

        return rewards, final_reviews

class MajorityVotingVerifier():
    """
    Runs multiple parallel reviews using the same checklist as Verifier.

    Chooses the majority verdict among the reviews as the final decision.
    Returns the first review whose verdict matches the majority as the response.
    Tie policy: randomly select one of the reviews and adopt its verdict and
    text as the final result.
    """
    def __init__(self, api_base, api_key, model, review_times: int = 3):
        self.client = LLMClient(api_base, api_key, model)
        self.review_times = max(1, review_times)

    def _review_messages(self, problems, completions):
        messages = []
        for (p, c) in zip(problems, completions):
            answer = strip_think_simple(c if isinstance(c, str) else c[0]['content'])
            base = [
                {"role": "system", "content": (
                    "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
                )},
                {"role": "user", "content": (
                    "Here is a math problem and a candidate solution of it, and you need to verify the correctness of this solution. Please check each of the following:\n\n"
                    "1. The provided content is indeed a math problem and its corresponding solution, rather than unrelated material supplied by mistake.\n"
                    "2. The solution actually derives the conclusion required by the original problem.\n"
                    "3. Every step of calculation and formula derivation in the solution is correct.\n"
                    "4. The hypotheses (conditions) and conclusions of any theorems used are correctly matched and applied.\n"
                    "5. The solution relies only on the conditions given in the problem and does not introduce any additional assumptions to obtain the conclusion.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the solution as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness (e.g., invalid step, wrong theorem usage without required conditions, uncorrected calculation error leading to a wrong result), treat the solution as incorrect.\n\n"
                    "Response requirements: If the solution is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any."
                    " If the solution is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error."
                    " Do not include any restatement of the entire solution or problem.\n\n"
                    f"<problem>{p}</problem>\n\n"
                    f"<answer>{answer}</answer>"
                )}
            ]
            for _ in range(self.review_times):
                messages.append(base)
        return messages

    def __call__(self, problems, completions, **kwargs):
        review_messages = self._review_messages(problems, completions)
        all_reviews = ASYNC_LOOP.run(self.client.infer_batch_async(review_messages, **kwargs))
        k = self.review_times
        grouped = [all_reviews[i * k:(i + 1) * k] for i in range(len(problems))]

        final_reviews = []
        rewards = []
        for reviews in grouped:
            # Count verdicts
            positives = 0
            negatives = 0
            for r in reviews:
                verdict = extract_xml_content(r, "verification")
                if verdict == "true":
                    positives += 1
                else:
                    negatives += 1

            # Determine majority verdict; if tie, randomly choose a review
            if positives > negatives:
                majority_verdict_true = True
                # Pick the first review that matches the majority verdict
                chosen = None
                for r in reviews:
                    if extract_xml_content(r, "verification") == "true":
                        chosen = r
                        break
                chosen = chosen or (reviews[0] if reviews else "")
            elif negatives > positives:
                majority_verdict_true = False
                chosen = None
                for r in reviews:
                    if extract_xml_content(r, "verification") == "false":
                        chosen = r
                        break
                chosen = chosen or (reviews[0] if reviews else "")
            else:
                # Tie: randomly select one review as the final result
                if reviews:
                    chosen = random.choice(reviews)
                    majority_verdict_true = extract_xml_content(chosen, "verification") == "true"
                else:
                    chosen = ""
                    majority_verdict_true = False

            rewards.append(1.0 if majority_verdict_true else 0.0)
            final_reviews.append(chosen)

        return rewards, final_reviews

class VPessimisticVerifier():
    """
    Chunked pessimistic verifier.

    Instead of reviewing the whole proof at once, it splits the proof into
    chunks of `chunk_length` lines. For each chunk, it asks the reviewer to
    focus only on that chunk while still providing the full problem and full
    proof for context. If any chunk is flagged incorrect (`<verification>false</verification>`),
    the final verdict is false. It also aggregates all error reports found.
    """
    def __init__(self, api_base, api_key, model, chunk_length: int = 7):
        self.client = LLMClient(api_base, api_key, model)
        self.chunk_length = max(1, int(chunk_length))
        
        # Constant fallback text when no critical errors are found across all chunks
        self.NO_ERROR_FALLBACK: str = (
            "<verification>true</verification>\n"
            "No critical error found in this proof after chunked review. "
            "All inspected chunks were considered correct overall given the problem and prior steps. "
            "Minor, non-decisive issues (e.g., superficial notation or small slips later corrected) "
            "may exist but do not undermine correctness."
        )

    def _split_into_chunks(self, proof: str) -> list[str]:
        lines = (proof or "").splitlines()
        chunks = []
        for i in range(0, len(lines), self.chunk_length):
            chunk_lines = lines[i:i + self.chunk_length]
            chunks.append("\n".join(chunk_lines))
        if not chunks:
            chunks = [proof or ""]
        return chunks

    def _build_messages_for_one(self, problem: str, full_proof: str) -> list[list[dict]]:
        """Build messages for all chunks of a single (problem, proof)."""
        chunks = self._split_into_chunks(full_proof)
        messages_per_chunk = []
        for idx, chunk in enumerate(chunks, start=1):
            messages_per_chunk.append([
                {"role": "system", "content": (
                    "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
                )},
                {"role": "user", "content": (
                    "We provide the original problem and the complete proposed solution for full context. "
                    "Then we provide a specific chunk from the solution for focused checking. "
                    "Your task: Check ONLY the given chunk for errors while considering the overall context.\n\n"
                    "Checklist:\n"
                    "1. The chunk’s reasoning and calculations adhere to mathematical correctness.\n"
                    "2. Any theorems used in the chunk match their hypotheses and conclusions.\n"
                    "3. The chunk does not rely on assumptions not justified by the problem or earlier proven steps.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the chunk as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness in this chunk (e.g., invalid step, wrong theorem usage without required conditions), treat the chunk as incorrect.\n\n"
                    "Response requirements: If the chunk is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any. "
                    "If the chunk is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error in the chunk.\n\n"
                    f"<problem>{problem}</problem>\n\n"
                    f"<full_answer>{strip_think_simple(full_proof)}</full_answer>\n\n"
                    f"<chunk_index>{idx}</chunk_index>\n"
                    f"<chunk>{chunk}</chunk>"
                )}
            ])
        return messages_per_chunk

    def __call__(self, problems, completions, **kwargs):
        """
        For each proof, review all chunks. Any chunk error makes the final verdict false.
        Returns evals (1.0 or 0.0 per proof) and aggregated review texts per proof.
        """
        # Build all chunk messages across the batch
        batch_messages = []
        per_item_chunk_counts = []
        for p, c in zip(problems, completions):
            full_answer = c if isinstance(c, str) else c[0]['content']
            full_answer = strip_think_simple(full_answer)
            msgs = self._build_messages_for_one(p, full_answer)
            per_item_chunk_counts.append(len(msgs))
            batch_messages.extend(msgs)

        # Expose counts for logging/analysis
        self.last_chunk_counts = per_item_chunk_counts[:]

        # Run inference over all chunks
        all_chunk_reviews = ASYNC_LOOP.run(self.client.infer_batch_async(batch_messages, **kwargs))

        # Group reviews by original sample
        grouped_reviews = []
        cursor = 0
        for count in per_item_chunk_counts:
            grouped_reviews.append(all_chunk_reviews[cursor:cursor + count])
            cursor += count

        # Aggregate verdicts and collect all errors
        evals = []
        final_texts = []
        for reviews in grouped_reviews:
            has_error = False
            errors_text = []
            fallback_text = reviews[0] if reviews else ""
            for r in reviews:
                verdict = extract_xml_content(r, "verification")
                if verdict == "false":
                    has_error = True
                    errors_text.append(strip_think_simple(r))
            if has_error:
                evals.append(0.0)
                # Aggregate all error reports into one text block
                combined = "\n\n".join(errors_text) if errors_text else fallback_text
                final_texts.append(combined)
            else:
                evals.append(1.0)
                # If no errors, return a constant message instead of first review
                final_texts.append(self.NO_ERROR_FALLBACK)

        return evals, final_texts

class ProgressivePessimisticVerifier():
    """
    Iteratively applies chunked pessimistic verification with progressively
    finer granularity. It starts with a full-proof check and then doubles the
    number of chunks (down to min_chunk_size per chunk) for still-positive
    samples until either an error is found or max_iters is reached.
    """
    def __init__(self, api_base, api_key, model, max_iters: int = 3, min_chunk_size: int = 6):
        self.client = LLMClient(api_base, api_key, model)
        self.max_iters = max(1, int(max_iters))
        self.min_chunk_size = max(1, int(min_chunk_size))
        self.last_review_counts: list[int] = []

        self.NO_ERROR_FALLBACK: str = (
            "<verification>true</verification>\n"
            "No critical error found in this proof after progressive chunked review. "
            "All passes (from coarse to fine) considered the solution correct overall. "
            "Minor, non-decisive issues may exist but do not undermine correctness."
        )

    def _split_into_chunks(self, proof: str, chunk_length: int) -> list[str]:
        lines = (proof or "").splitlines()
        if not lines:
            return [proof or ""]
        chunks = []
        for i in range(0, len(lines), chunk_length):
            chunk_lines = lines[i:i + chunk_length]
            chunks.append("\n".join(chunk_lines))
        return chunks

    def _build_messages_for_one(self, problem: str, full_proof: str, chunk_length: int) -> list[list[dict]]:
        chunks = self._split_into_chunks(full_proof, chunk_length)
        messages_per_chunk = []
        for idx, chunk in enumerate(chunks, start=1):
            messages_per_chunk.append([
                {"role": "system", "content": (
                    "You are an assistant highly proficient in mathematics. The user will provide a math problem together with its proposed solution, and your task is to verify the correctness of that solution according to the given instruction."
                )},
                {"role": "user", "content": (
                    "We provide the original problem and the complete proposed solution for full context. "
                    "Then we provide a specific chunk from the solution for focused checking. "
                    "Your task: Check ONLY the given chunk for errors while considering the overall context.\n\n"
                    "Checklist:\n"
                    "1. The chunk’s reasoning and calculations adhere to mathematical correctness.\n"
                    "2. Any theorems used in the chunk match their hypotheses and conclusions.\n"
                    "3. The chunk does not rely on assumptions not justified by the problem or earlier proven steps.\n\n"
                    "Consistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the chunk as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness in this chunk (e.g., invalid step, wrong theorem usage without required conditions), treat the chunk as incorrect.\n\n"
                    "Response requirements: If the chunk is correct overall (possibly with minor issues), reply with `<verification>true</verification>` and briefly list minor issues if any. "
                    "If the chunk is incorrect, reply with `<verification>false</verification>` followed by a concise description of the most harmful error in the proof that you found in the chunk.\n\n"
                    f"<problem>{problem}</problem>\n\n"
                    f"<full_answer>{full_proof}</full_answer>\n\n"
                    f"<chunk_index>{idx}</chunk_index>\n"
                    f"<chunk>{chunk}</chunk>"
                )}
            ])
        return messages_per_chunk

    def _chunk_length_for_iteration(self, proof: str, iteration: int) -> int:
        lines = (proof or "").splitlines()
        num_lines = len(lines)
        if num_lines == 0:
            return self.min_chunk_size
        if iteration == 0:
            return max(num_lines, self.min_chunk_size)
        target_chunks = max(1, 2 ** iteration)
        approx_length = (num_lines + target_chunks - 1) // target_chunks
        return max(self.min_chunk_size, approx_length)

    def __call__(self, problems, completions, **kwargs):
        total = len(problems)
        if total == 0:
            self.last_review_counts = []
            return [], []

        proofs = [strip_think_simple(c if isinstance(c, str) else c[0]['content']) for c in completions]
        evals: list[float | None] = [None] * total
        final_texts = [""] * total
        pending_indices = list(range(total))
        total_review_counts = [0] * total

        for iteration in range(self.max_iters):
            if not pending_indices:
                break

            batch_messages = []
            per_item_counts = []
            index_order = []
            for idx in pending_indices:
                problem = problems[idx]
                proof = proofs[idx]
                chunk_length = self._chunk_length_for_iteration(proof, iteration)
                msgs = self._build_messages_for_one(problem, proof, chunk_length)
                index_order.append(idx)
                per_item_counts.append(len(msgs))
                total_review_counts[idx] += len(msgs)
                batch_messages.extend(msgs)

            if not batch_messages:
                break

            chunk_reviews = ASYNC_LOOP.run(self.client.infer_batch_async(batch_messages, **kwargs))

            cursor = 0
            next_pending = []
            for sample_idx, count in zip(index_order, per_item_counts):
                sample_reviews = chunk_reviews[cursor:cursor + count]
                cursor += count

                chunk_errors: list[str] = []
                for chunk_id, review in enumerate(sample_reviews, start=1):
                    verdict = extract_xml_content(review, "verification")
                    if verdict == "false":
                        formatted = strip_think_simple(review)
                        chunk_errors.append(f"[chunk {chunk_id}] {formatted}")

                if chunk_errors:
                    evals[sample_idx] = 0.0
                    final_texts[sample_idx] = "\n\n".join(chunk_errors)
                else:
                    if iteration == self.max_iters - 1:
                        evals[sample_idx] = 1.0
                        final_texts[sample_idx] = self.NO_ERROR_FALLBACK
                    else:
                        next_pending.append(sample_idx)

            pending_indices = next_pending

        # Any remaining samples (e.g., no further iterations but never failed) are treated as passes.
        for idx in pending_indices:
            if evals[idx] is None:
                evals[idx] = 1.0
                final_texts[idx] = self.NO_ERROR_FALLBACK

        # For any sample that never received a review (e.g., empty proof), ensure defaults.
        for i, value in enumerate(evals):
            if value is None:
                evals[i] = 1.0
                final_texts[i] = self.NO_ERROR_FALLBACK

        self.last_review_counts = total_review_counts
        return evals, final_texts

class NaiveProver():
    """
    NaiveProver directly proves the given problem
    """
    def __init__(self, api_base, api_key, model):
        self.client = LLMClient(api_base, api_key, model)

    def __call__(self, problems: list[str], **kwargs):
        all_messages = [
            [
                {"role": "user", "content": f"Please provide a complete and rigorous solution to this problem:\n\n{p}"}
            ]
            for p in problems
        ]
        results = ASYNC_LOOP.run(self.client.infer_batch_async(all_messages, **kwargs))
        return results

def main():
    parser = argparse.ArgumentParser(
        description="RProver"
    )
    parser.add_argument("-ed", "--eval_dataset", help="the path to the dataset used for evaluation", default="")
    parser.add_argument("-pm", "--proof_model", help="model that generates proofs for given problems", default="")
    parser.add_argument("-em", "--eval_model", help="the model used for evaluation (if needed)", default="")
    parser.add_argument("-gm", "--guider_model", help="the model used for guidance", default="")
    parser.add_argument("--log_dir", help="the logging directory path", default="eval_logs")
    parser.add_argument("--reasoning_effort", help="the reasoning_effort parameter for some models", default="medium", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--method", default="naive", choices=["naive"], help="the training / evaluation method switch")
    parser.add_argument("--reviewer", default="standard", choices=["standard", "pessimistic", "pessimistic_judger", "vpessimistic", "majority", "progressive"], help="the reviewer used for evaluation")
    parser.add_argument("--reviews", type=int, default=3, help="number of parallel reviews for multi-review verifiers (pessimistic/majority)")
    parser.add_argument("--chunk_length", type=int, default=7, help="lines per chunk for vpessimistic reviewer")
    parser.add_argument("--progressive_max_iters", type=int, default=3, help="maximum refinement passes for progressive reviewer")
    parser.add_argument("--progressive_min_chunk_size", type=int, default=6, help="minimum lines per chunk for progressive reviewer")
    parser.add_argument("--evaluate_reviewer", action='store_true', default=False, help="enable evaluation of the reviewer against guider model as ground truth")
    parser.add_argument("--prover_base_url", default="", help="the base url for prover")
    parser.add_argument("--eval_base_url", default="", help="the base url for evaluator")
    parser.add_argument("--guider_base_url", default="", help="the base url for guider model (falls back to eval -> prover)")
    parser.add_argument("--prover_api_key", default="", help="the api key for the prover")
    parser.add_argument("--eval_api_key", default="", help="the api key for the evaluator")
    parser.add_argument("--guider_api_key", default="", help="the api key for the guider model (falls back to eval -> prover)")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True, help="toggle enable_thinking parameter for models that support reasoning traces")
    parser.add_argument("--agenttrain", action='store_true', default=False, help="enable agentic training while running this program")
    parser.add_argument(
        "--verifier_samples",
        default="",
        help=(
            "path to a previously generated verifier_samples.json. "
            "When set, uses the same problems, proofs, and golden labels from the file, "
            "skipping new proof generation and ground-truth verification."
        ),
    )

    logger = logging.getLogger("main")
    args = parser.parse_args()
    logger.info("start verifying with proof_model: %s", args.proof_model)
    logger.info("using eval model: %s", args.eval_model)

    # If verifier_samples is provided, use it to load problems/proofs and GT labels
    loaded_verifier_samples = None
    if args.verifier_samples:
        if args.verifier_samples == "Salesforce/Hard2Verify" or args.verifier_samples == "INSAIT-Institute/OPC" or args.verifier_samples == "NP_dataset/gradingbench.csv":
            ds = prepare_dataset(args.verifier_samples)
            problems = ds["problem"]
            proofs = ds["proof"]
            # preloaded_gt_texts = [e["human_labels_first_error_idx"] for e in ds]
            preloaded_gt_labels = ds["gt_eval"]
            preloaded_gt_texts = [None] * len(problems)
            logger.info("Loaded %d samples from verifier_samples: %s", len(problems), args.verifier_samples)
        else:
            vs_path = Path(args.verifier_samples)
            with vs_path.open("r", encoding="utf-8") as f:
                loaded_verifier_samples = json.load(f)
            if not isinstance(loaded_verifier_samples, list):
                raise ValueError("verifier_samples must be a list of sample dicts")
            problems = [s.get("problem", "") for s in loaded_verifier_samples]
            proofs = [s.get("proof", "") for s in loaded_verifier_samples]
            preloaded_gt_labels = [bool(s.get("gt_label", False)) for s in loaded_verifier_samples]
            preloaded_gt_texts = [s.get("gt_text", "") for s in loaded_verifier_samples]
            logger.info("Loaded %d samples from verifier_samples: %s", len(problems), vs_path)
    else:
        ds = prepare_dataset(args.eval_dataset)
        problems = [e['problem'] for e in ds]

    # Resolve API bases and keys with fallback: prover -> eval -> guider
    prover_base_url = args.prover_base_url
    prover_api_key = args.prover_api_key

    eval_base_url = args.eval_base_url or prover_base_url
    eval_api_key = args.eval_api_key or prover_api_key

    guider_base_url = args.guider_base_url or eval_base_url
    guider_api_key = args.guider_api_key or eval_api_key

    prover = NaiveProver(
        api_base=prover_base_url,
        api_key=prover_api_key,
        model=args.proof_model,
    )

    logdir = get_current_log_path(args.log_dir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Collect proofs unless verifier_samples is provided
    if args.verifier_samples:
        striped_proofs = [strip_think_simple(proof) for proof in proofs]
        logger.info("Using preloaded proofs from verifier_samples, skipping prover generation")
    else:
        proofs = prover(
            problems,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
        )
        striped_proofs = [strip_think_simple(proof) for proof in proofs]
        logger.info("successfully collected %d proofs from %s", len(proofs), args.proof_model)

    if args.reviewer == "pessimistic":
        # Use the new PessimisticVerifier (first error wins)
        evaluator = PessimisticVerifier(eval_base_url, eval_api_key, args.eval_model, review_times=args.reviews)
    elif args.reviewer == "majority":
        # Majority voting over multiple reviews
        evaluator = MajorityVotingVerifier(eval_base_url, eval_api_key, args.eval_model, review_times=args.reviews)
    elif args.reviewer == "vpessimistic":
        # Chunked pessimistic verifier (focus per-chunk)
        evaluator = VPessimisticVerifier(eval_base_url, eval_api_key, args.eval_model, chunk_length=args.chunk_length)
    elif args.reviewer == "progressive":
        # Progressive chunking: coarse-to-fine pessimistic verifier
        evaluator = ProgressivePessimisticVerifier(
            eval_base_url,
            eval_api_key,
            args.eval_model,
            max_iters=args.progressive_max_iters,
            min_chunk_size=args.progressive_min_chunk_size,
        )
    elif args.reviewer == "pessimistic_judger":
        # Two-phase: parallel reviews then final judger decision
        evaluator = PessimisticJudger(eval_base_url, eval_api_key, args.eval_model, review_times=args.reviews)
    else:
        evaluator = Verifier(eval_base_url, eval_api_key, args.eval_model)
    evals, verifications = evaluator(
        problems,
        striped_proofs,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=args.enable_thinking,
    )
    accuracy = sum(evals) / len(evals)
    logger.info(f"Obtained final accuracy: {accuracy}")

    # Optional: evaluate the reviewer against the guider model as ground truth
    if args.evaluate_reviewer:
        logger.info("Evaluating reviewer against guider model as ground truth")
        if args.verifier_samples:
            # Use ground-truth labels/texts from the provided verifier_samples file
            gt_labels = preloaded_gt_labels
            gt_texts = preloaded_gt_texts
            logger.info("Using GT labels from verifier_samples; skipping new GT verification")
        else:
            gt_verifier = ProgressivePessimisticVerifier(guider_base_url,
                                                         guider_api_key,
                                                         args.guider_model,
                                                         max_iters=args.progressive_max_iters,
                                                         min_chunk_size=args.progressive_min_chunk_size,
                                                         )
            gt_labels, gt_texts = gt_verifier(
                problems,
                striped_proofs,
                reasoning_effort=args.reasoning_effort,
                enable_thinking=args.enable_thinking,
            )

        preds = [int(x) for x in evals]
        gts = [int(x) for x in gt_labels]
        total = len(preds)
        correct = sum(1 for p, g in zip(preds, gts) if p == g)
        tp = sum(1 for p, g in zip(preds, gts) if p == 1 and g == 1)
        tn = sum(1 for p, g in zip(preds, gts) if p == 0 and g == 0)
        fp = sum(1 for p, g in zip(preds, gts) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(preds, gts) if p == 0 and g == 1)
        verifier_accuracy = correct / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall)) else None

        verifier_eval = {
            "total": total,
            "accuracy": verifier_accuracy,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        # Save sample-level comparison
        verifier_samples = [
            {
                "problem": problem,
                "proof": proof,
                "pred_label": bool(pred),
                "pred_text": pred_text,
                "gt_label": bool(gt),
                "gt_text": gt_text,
            }
            for (problem, proof, pred, pred_text, gt, gt_text) in zip(
                problems, proofs, preds, verifications, gts, gt_texts
            )
        ]

        with open(logdir / "verifier_eval.json", "w", encoding="utf-8") as f:
            json.dump(verifier_eval, f, ensure_ascii=False, indent=2, default=str)
        with open(logdir / "verifier_samples.json", "w", encoding="utf-8") as f:
            json.dump(verifier_samples, f, ensure_ascii=False, indent=2, default=str)

        # Add summary to logs.json payload
        vars_dict_key = "verifier_evaluation"
        # vars_dict defined below; collect into a temporary dict for later merge
        extra_verifier_eval = {vars_dict_key: verifier_eval}


    logger.info("evaluation ended")
    vars_dict = vars(args)
    vars_dict["accuracy"] = accuracy
    # Reviewer cost metrics for post-hoc cost/performance analysis
    reviewer_cost = {"reviewer": args.reviewer}
    num_samples = len(problems)
    if args.reviewer in {"vpessimistic", "progressive"}:
        if args.reviewer == "vpessimistic":
            counts = getattr(evaluator, "last_chunk_counts", []) or []
        else:
            counts = getattr(evaluator, "last_review_counts", []) or []
        total_reviews = sum(counts)
        avg_per_sample = (total_reviews / len(counts)) if counts else 0.0
        reviewer_cost.update({
            "total_reviews": total_reviews,
            "avg_reviews_per_sample": avg_per_sample,
            "min_reviews_per_sample": (min(counts) if counts else 0),
            "max_reviews_per_sample": (max(counts) if counts else 0),
        })
    else:
        if args.reviewer == "standard":
            per_sample = 1
        elif args.reviewer in {"pessimistic", "majority"}:
            per_sample = int(args.reviews)
        elif args.reviewer == "pessimistic_judger":
            per_sample = int(args.reviews) + 1  # k reviews + 1 final judger
        else:
            per_sample = 1
        reviewer_cost.update({
            "reviews_per_sample": per_sample,
            "total_reviews": per_sample * num_samples,
        })
    vars_dict["reviewer_cost"] = reviewer_cost
    # Token stats: skip prover token stats when using preloaded samples
    if args.verifier_samples:
        average_prover_inp_tokens = None
        average_prover_opt_tokens = None
    else:
        average_prover_inp_tokens = (
            sum(prover.client.input_tokens) / len(prover.client.input_tokens)
            if prover.client.input_tokens else 0.0
        )
        average_prover_opt_tokens = (
            sum(prover.client.comp_tokens) / len(prover.client.comp_tokens)
            if prover.client.comp_tokens else 0.0
        )
    average_eval_inp_tokens = (
        sum(evaluator.client.input_tokens) / len(evaluator.client.input_tokens)
        if evaluator.client.input_tokens else 0.0
    )
    average_eval_opt_tokens = (
        sum(evaluator.client.comp_tokens) / len(evaluator.client.comp_tokens)
        if evaluator.client.comp_tokens else 0.0
    )
    logger.info(f"Average token inputs in prover: {average_prover_inp_tokens}")
    logger.info(f"Average completion tokens in prover: {average_prover_opt_tokens}")
    logger.info(f"Average token inputs in evaluator: {average_eval_inp_tokens}")
    logger.info(f"Average completion inputs in evaluator: {average_eval_opt_tokens}")
    vars_dict["average_prover_inp_tokens"] = average_prover_inp_tokens
    vars_dict["average_prover_opt_tokens"] = average_prover_opt_tokens
    vars_dict["average_eval_inp_tokens"] = average_eval_inp_tokens
    vars_dict["average_eval_opt_tokens"] = average_eval_opt_tokens

    # Merge verifier evaluation summary if available
    try:
        if 'extra_verifier_eval' in locals():
            vars_dict.update(extra_verifier_eval)
    except Exception:
        pass

    with open(logdir / "logs.json", "w", encoding="utf-8") as f:
        json.dump(vars_dict, f, ensure_ascii=False, indent=2, default=str)

    # Prepare sample payload; use placeholder tokens if verifier_samples is provided
    if args.verifier_samples:
        prover_inp_tokens = [None] * len(problems)
        prover_comp_tokens = [None] * len(problems)
    else:
        prover_inp_tokens = prover.client.last_input_tokens
        prover_comp_tokens = prover.client.last_comp_tokens

    samples = [
        {
            "problem": problem,
            "proof": proof,
            "eval": eval,
            "verification": verification,
            "input_tokens": inp_tokens,
            "completion_tokens": comp_tokens
        }
        for (problem, proof, eval, verification, inp_tokens, comp_tokens) in zip(
            problems, proofs, evals, verifications, prover_inp_tokens, prover_comp_tokens
        )
    ]
    # if exps is not None:
    #     for s, exp in zip(samples, exps):
    #         s["exp"] = exp

    with open(logdir / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"successfully saved logs to path {logdir}")

if __name__ == "__main__":
    LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FMT,
        datefmt=DATE_FMT,
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info("Program Started")
    main()
