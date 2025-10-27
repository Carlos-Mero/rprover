import argparse
import json
import asyncio
import re
from pathlib import Path
from utils.async_runner import AsyncLoopThread
from datasets import load_dataset, Dataset, concatenate_datasets
from litellm import acompletion
import httpx
import logging
from datetime import datetime, timezone
import random
import requests

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
    else:
        raise NotImplementedError(f"Unknown dataset name or path: {dataset_path}")

    logger.info("completed preparing dataset at: %s", dataset_path)

    return ds

class LightRAGClient():
    def __init__(self, base_url):
        self.url = base_url
        self.logger = logging.getLogger("LightRAGClient")

    def insert(self, text: str, source=None):
        # insert text to the rag server
        url = self.url + "/documents/text"
        data = {"text": text}
        if source is not None:
            data["file_source"] = source
        try:
            response = requests.post(url, json=data).json()
            self.logger.info("%s", response["message"])
        except requests.exceptions.RequestException as e:
            self.logger.error("Requests error occured: %s", e)

    def insert_texts(self, texts: list[str], sources=None):
        url = self.url + "/documents/texts"
        data = {"texts": texts}
        if sources is not None:
            data["file_sources"] = sources
        try:
            response = requests.post(url, json=data).json()
            self.logger.info("%s", response["message"])
        except requests.exceptions.RequestException as e:
            self.logger.error("Requests error occured: %s", e)

    def query(self, prompt):
        url = self.url + "/query"
        data = {"query": prompt, "mode": "mix", "enable_rerank": False}
        try:
            response = requests.post(url, json=data).json()
            return response, response.get("references")
        except requests.exceptions.RequestException as e:
            self.logger.error("Requests error occured: %s", e)
            return f"Error occured on query: {e}", None

    async def aquery(self, prompt: str):
        url = self.url + "/query"
        data = {"query": prompt, "mode": "mix", "enable_rerank": False}
        timeout = httpx.Timeout(connect=5.0, read=3600.0, write=30.0, pool=5.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=data)
                response_json = response.json()
                return response_json, response_json.get("references")
        except httpx.RequestError as e:
            self.logger.error("HTTPX request error occured: %s", e)
            return None

    async def async_batch_query(self, prompts: list[str]):
        # sem = asyncio.Semaphore(concurrency)
        tasks = [
            asyncio.create_task(self.aquery(prompt))
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                raise RuntimeError(f"Task {i} failed") from r
        logger.info("completed batch rag query on %d samples",  len(prompts))
        return results

class LLMClient():
    def __init__(self, api_base, api_key, model):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.input_tokens = []
        self.comp_tokens = []

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
                                **kwargs) -> list[str]:
        logger = logging.getLogger("evaluator")
        logger.info("running batch inference on %d samples", len(all_messages))
        sem = asyncio.Semaphore(concurrency)
        ALLOWED_PARAM_KEYS = {"reasoning_effort", "thinking", "enable_thinking"}
        infer_params = {k: v for k, v in kwargs.items() if k in ALLOWED_PARAM_KEYS}
        tasks = [
            asyncio.create_task(self._infer_one(messages, sem, **infer_params))
            for messages in all_messages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                raise RuntimeError(f"Task {i} failed") from r
        logger.info("completed batch inference on %d samples",  len(all_messages))
        completions = [r.choices[0].message["content"] if r is not None else "" for r  in results]
        self.input_tokens = [r.usage.prompt_tokens for r in results if r is not None]
        self.comp_tokens = [r.usage.completion_tokens for r in results if r is not None]
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
                    "Here is a math problem and a candidate solution of it, and you need to verify the correctness of this solution. Please check each of the following:\n"
                    "\n"
                    "1. The provided content is indeed a math problem and its corresponding solution, rather than unrelated material supplied by mistake.\n"
                    "2. The solution actually derives the conclusion required by the original problem.\n"
                    "3. Every step of calculation and formula derivation in the solution is correct.\n"
                    "4. The hypotheses (conditions) and conclusions of any theorems used are correctly matched and applied.\n"
                    "5. The solution relies only on the conditions given in the problem and does not introduce any additional assumptions to obtain the conclusion.\n"
                    "\nConsistency and error-severity policy (important):\n"
                    "- If only minor, easily fixable issues exist (e.g., small algebraic slips later corrected, notational typos, superficial formatting), treat the solution as correct overall but briefly note such issues.\n"
                    "- If there is any critical error that undermines correctness (e.g., invalid step, wrong theorem usage without required conditions, uncorrected calculation error leading to a wrong result), treat the solution as incorrect.\n"
                    "\n"
                    "If correct overall (possibly with minor issues), append `<verification>true</verification>` at the end of your reply and include a brief note of minor issues if any; otherwise append `<verification>false</verification>`.\n"
                    "\n"
                    f"<problem>{p}</problem>\n"
                    "\n"
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

class GProver():
    """
    This prover directly receives guidance from a stronger gmodel and tries to prove a new theorem
    """
    def __init__(self, api_base, api_key, model, gapi_base, gapi_key, gmodel):
        self.client = LLMClient(api_base, api_key, model)
        self.gclient = LLMClient(gapi_base, gapi_key, gmodel)

    def __call__(self, problems: list[str], **kwargs):
        g_messages = [
            [
                {"role": "user", "content": f"Please provide a complete and rigorous solution to this problem:\n\n{p}"}
            ]
            for p in problems
        ]
        gresults = ASYNC_LOOP.run(self.gclient.infer_batch_async(g_messages, **kwargs))
        all_messages = [
            [
                {"role": "user", "content": f"Please provide a complete and rigorous solution to this problem:\n\n{p}\n\nHere we have collected a candidate proof which you may use as a reference:\n\n{proof}"}
            ]
            for p, proof in zip(problems, gresults)
        ]
        results = ASYNC_LOOP.run(self.client.infer_batch_async(all_messages, **kwargs))
        return results

class HProver():
    """
    This prover tries to prove a problem with the hints from a stronger LRM
    """
    def __init__(self, api_base, api_key, model, gapi_base, gapi_key, gmodel):
        self.client = LLMClient(api_base, api_key, model)
        self.gclient = LLMClient(gapi_base, gapi_key, gmodel)

    def __call__(self, problems: list[str], **kwargs):
        g_messages = [
            [
                {"role": "user", "content": f"Please carefully analyse this problem and provide some hints on how to solve it:\n\n{p}"}
            ]
            for p in problems
        ]
        gresults = ASYNC_LOOP.run(self.gclient.infer_batch_async(g_messages, **kwargs))
        all_messages = [
            [
                {"role": "user", "content": f"Please provide a complete and rigorous solution to this problem:\n\n{p}\n\nHere are some hints that might help you with this problem:\n\n{hint}"}
            ]
            for p, hint in zip(problems, gresults)
        ]
        results = ASYNC_LOOP.run(self.client.infer_batch_async(all_messages, **kwargs))
        return results

class ACEProver():
    """
    This prover collects a series of hints from gpt-5 and tries to prove new theorems
    """
    def __init__(self, api_base, api_key, model, gapi_base, gapi_key, gmodel):
        self.client = LLMClient(api_base, api_key, model)
        self.gclient = LLMClient(gapi_base, gapi_key, gmodel)
        self.exps = []

    def format_exps(self):
        all_exp = ""
        for i, exp in enumerate(self.exps):
            all_exp += f"{i}. {exp}"
        return all_exp

    def __call__(self, problems: list[str], **kwargs):
        g_messages = [
            [
                {"role": "user", "content": f"Please carefully analyse this problem and summarize some useful experiences that might be useful for solving similar problems. You should respond in no more than two sentences:\n\n{p}"}
            ]
            for p in problems[:100]
        ]
        gresults = ASYNC_LOOP.run(self.gclient.infer_batch_async(g_messages, **kwargs))
        self.exps = gresults
        full_exp = self.format_exps()

        all_messages = [
            [
                {"role": "user", "content": f"You are an experienced mathematician that is required to solve a math problem. Here are some instructions that might help you with math problems:\n\n{full_exp}\n\nNow please provide a complete and rigorous solution to this problem:\n\n{p}"}
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
    parser.add_argument("--method", default="rlvr", choices=["naive", "gprover", "hprover", "aceprover"], help="the training / evaluation method switch")
    parser.add_argument("--reviewer", default="standard", choices=["standard", "pessimistic", "pessimistic_judger"], help="the reviewer used for evaluation")
    parser.add_argument("--reviews", type=int, default=3, help="number of parallel reviews for pessimistic reviewer")
    parser.add_argument("--evaluate_reviewer", action='store_true', default=False, help="enable evaluation of the reviewer against guider model as ground truth")
    parser.add_argument("--prover_base_url", default="", help="the base url for prover")
    parser.add_argument("--eval_base_url", default="", help="the base url for evaluator")
    parser.add_argument("--guider_base_url", default="", help="the base url for guider model (falls back to eval -> prover)")
    parser.add_argument("--prover_api_key", default="", help="the api key for the prover")
    parser.add_argument("--eval_api_key", default="", help="the api key for the evaluator")
    parser.add_argument("--guider_api_key", default="", help="the api key for the guider model (falls back to eval -> prover)")
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

    if args.method == "naive":
        prover = NaiveProver(
            api_base=prover_base_url,
            api_key=prover_api_key,
            model=args.proof_model,
        )
    elif args.method == "gprover":
        prover = GProver(
            api_base=prover_base_url,
            api_key=prover_api_key,
            model=args.proof_model,
            gapi_base=guider_base_url,
            gapi_key=guider_api_key,
            gmodel=args.guider_model
        )
    elif args.method == "hprover":
        prover = HProver(
            api_base=prover_base_url,
            api_key=prover_api_key,
            model=args.proof_model,
            gapi_base=guider_base_url,
            gapi_key=guider_api_key,
            gmodel=args.guider_model
        )
    elif args.method == "aceprover":
        prover = ACEProver(
            api_base=prover_base_url,
            api_key=prover_api_key,
            model=args.proof_model,
            gapi_base=guider_base_url,
            gapi_key=guider_api_key,
            gmodel=args.guider_model
        )
    elif args.method == "aceprover":
        raise NotImplementedError("aceprover is not implemented")
    else:
        raise NotImplementedError("unknown method")

    logdir = get_current_log_path(args.log_dir)
    logdir.mkdir(parents=True, exist_ok=True)

    # Collect proofs unless verifier_samples is provided
    if args.verifier_samples:
        striped_proofs = [strip_think_simple(proof) for proof in proofs]
        logger.info("Using preloaded proofs from verifier_samples, skipping prover generation")
    else:
        proofs = prover(problems, reasoning_effort=args.reasoning_effort)
        striped_proofs = [strip_think_simple(proof) for proof in proofs]
        logger.info("successfully collected %d proofs from %s", len(proofs), args.proof_model)

    if args.method == "naive" or args.method == "gprover" or args.method == "hprover" or args.method == "aceprover":
        if args.reviewer == "pessimistic":
            # Use the new PessimisticVerifier (first error wins)
            evaluator = PessimisticVerifier(eval_base_url, eval_api_key, args.eval_model, review_times=args.reviews)
        elif args.reviewer == "pessimistic_judger":
            # Two-phase: parallel reviews then final judger decision
            evaluator = PessimisticJudger(eval_base_url, eval_api_key, args.eval_model, review_times=args.reviews)
        else:
            evaluator = Verifier(eval_base_url, eval_api_key, args.eval_model)
        evals, verifications = evaluator(problems, striped_proofs, reasoning_effort=args.reasoning_effort)
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
                gt_verifier = Verifier(guider_base_url, guider_api_key, args.guider_model)
                gt_labels, gt_texts = gt_verifier(problems, striped_proofs, reasoning_effort=args.reasoning_effort)

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
    else:
        raise NotImplementedError("unknown method")


    logger.info("evaluation ended")
    vars_dict = vars(args)
    vars_dict["accuracy"] = accuracy
    # Token stats: skip prover token stats when using preloaded samples
    if args.verifier_samples:
        average_prover_inp_tokens = None
        average_prover_opt_tokens = None
    else:
        average_prover_inp_tokens = sum(prover.client.input_tokens) / len(prover.client.input_tokens)
        average_prover_opt_tokens = sum(prover.client.comp_tokens) / len(prover.client.comp_tokens)
    average_eval_inp_tokens = sum(evaluator.client.input_tokens) / len(evaluator.client.input_tokens)
    average_eval_opt_tokens = sum(evaluator.client.comp_tokens) / len(evaluator.client.comp_tokens)
    logger.info(f"Average token inputs in prover: {average_prover_inp_tokens}")
    logger.info(f"Average completion tokens in prover: {average_prover_opt_tokens}")
    logger.info(f"Average token inputs in evaluator: {average_eval_inp_tokens}")
    logger.info(f"Average completion inputs in evaluator: {average_eval_opt_tokens}")
    vars_dict["average_prover_inp_tokens"] = average_prover_inp_tokens
    vars_dict["average_prover_opt_tokens"] = average_prover_opt_tokens
    vars_dict["average_eval_inp_tokens"] = average_eval_inp_tokens
    vars_dict["average_eval_opt_tokens"] = average_eval_opt_tokens

    if hasattr(prover, 'exps'):
        vars_dict["exps"] = prover.exps

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
        prover_inp_tokens = prover.client.input_tokens
        prover_comp_tokens = prover.client.comp_tokens

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
