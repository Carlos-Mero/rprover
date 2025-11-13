import argparse
import asyncio
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from litellm import acompletion
from tqdm import tqdm

# Explicit API configuration for litellm calls
# Set these at the top of this file; no CLI overrides.
API_BASE = "https://api.openai.com/v1"
API_KEY = ""

STANDARD_RUBRIC = (
    "Here is a math problem and a candidate solution of it, and you need to verify the correctness of this solution. "
    "Please check each of the following:\n\n"
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
    " Do not include any restatement of the entire solution or problem."
)


def extract_xml_content(text: str, tag: str) -> str | None:
    pattern = rf"<{re.escape(tag)}(?:\s+[^>]*)?\s*>(.*?)</\s*{re.escape(tag)}\s*>"
    matches = list(re.finditer(pattern, text, flags=re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def strip_think_simple(s: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)


def load_verifier_samples(logdir: Path) -> List[Dict[str, Any]]:
    vs_path = logdir / "verifier_samples.json"
    if not vs_path.exists():
        raise FileNotFoundError(f"verifier_samples.json not found in {logdir}")
    with vs_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("verifier_samples.json must contain a list of samples")
    return samples


def parse_bool_tag(text: str | None) -> bool | None:
    if text is None:
        return None
    val = text.strip().lower()
    if val in {"true", "1", "yes"}:
        return True
    if val in {"false", "0", "no"}:
        return False
    return None


def normalize_issue_effect(label: str | None) -> str:
    lookup = {"critical", "minor", "unsupported"}
    if not label:
        return "unsupported"
    lower = label.strip().lower()
    return lower if lower in lookup else "unsupported"


def build_review_prompt(problem: str, proof: str, review: str) -> List[Dict[str, str]]:
    answer = strip_think_simple(proof)
    user_content = (
        f"{STANDARD_RUBRIC}\n\n"
        "You are also given a review that claims specific issues about the solution. "
        "Re-evaluate the solution yourself under the rubric above and determine whether the review's highlighted issue actually affects correctness. "
        "If the solution remains correct despite the review, explain briefly why the review's concern is non-critical. "
        "Return EXACTLY three XML tags in this order:\n"
        "<verification>true|false</verification> — your verdict on correctness under the rubric.\n"
        "<issue_effect>critical|minor|unsupported</issue_effect> — does the cited issue undermine correctness, note a minor issue, or fail to pinpoint a real issue.\n"
        "<explanation>...</explanation> — concise reasoning (max 4 sentences).\n\n"
        f"<problem>{problem}</problem>\n\n"
        f"<answer>{answer}</answer>\n\n"
        f"<review>{review}</review>"
    )

    return [
        {
            "role": "system",
            "content": (
                "You are an assistant highly proficient in mathematics. "
                "Follow the provided verification rubric exactly and keep responses concise."
            ),
        },
        {"role": "user", "content": user_content},
    ]


async def _reevaluate_one(sample: Dict[str, Any], sem: asyncio.Semaphore, model_name: str) -> Dict[str, Any]:
    problem = sample.get("problem", "")
    proof = sample.get("proof", "")
    review_text = sample.get("pred_text", "")
    orig_label = bool(sample.get("pred_label", False))

    messages = build_review_prompt(problem, proof, review_text)
    content = ""
    error: str | None = None
    try:
        async with sem:
            resp = await acompletion(
                model=model_name,
                messages=messages,
                api_base=API_BASE,
                api_key=API_KEY,
                drop_params=True,
            )
        content = resp.choices[0].message["content"] if resp is not None else ""
    except Exception as exc:  # pragma: no cover - best effort logging
        error = str(exc)

    verification = parse_bool_tag(extract_xml_content(content, "verification"))
    issue_effect = normalize_issue_effect(extract_xml_content(content, "issue_effect"))
    explanation = extract_xml_content(content, "explanation") or ""

    return {
        "problem": problem,
        "proof": proof,
        "review_text": review_text,
        "original_label": orig_label,
        "gt_label": bool(sample.get("gt_label", False)),
        "gt_text": sample.get("gt_text", ""),
        "reeval_response": content,
        "reeval_label": verification,
        "issue_effect": issue_effect,
        "reeval_explanation": explanation,
        "error": error,
    }


async def reevaluate_reviews(
    samples: List[Dict[str, Any]],
    model_name: str,
    concurrency: int = 8,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))
    pbar = tqdm(total=len(samples), desc="Review re-eval", leave=False) if show_progress else None

    async def _wrapped(sample: Dict[str, Any]):
        try:
            return await _reevaluate_one(sample, sem, model_name)
        finally:
            if pbar:
                pbar.update(1)

    tasks = [asyncio.create_task(_wrapped(sample)) for sample in samples]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if pbar:
            pbar.close()
    out: List[Dict[str, Any]] = []
    for sample, result in zip(samples, results):
        if isinstance(result, Exception):
            out.append(
                {
                    "problem": sample.get("problem", ""),
                    "proof": sample.get("proof", ""),
                    "review_text": sample.get("pred_text", ""),
                    "original_label": bool(sample.get("pred_label", False)),
                    "gt_label": bool(sample.get("gt_label", False)),
                    "gt_text": sample.get("gt_text", ""),
                    "reeval_response": "",
                    "reeval_label": None,
                    "issue_effect": "unsupported",
                    "reeval_explanation": "",
                    "error": str(result),
                }
            )
        else:
            out.append(result)
    return out


def compute_review_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    comparable = [r for r in results if r.get("reeval_label") is not None]
    skipped = total - len(comparable)

    preds = [bool(r.get("original_label", False)) for r in comparable]
    gts = [bool(r.get("reeval_label")) for r in comparable]

    tp = sum(1 for p, g in zip(preds, gts) if p and g)
    tn = sum(1 for p, g in zip(preds, gts) if not p and not g)
    fp = sum(1 for p, g in zip(preds, gts) if p and not g)
    fn = sum(1 for p, g in zip(preds, gts) if not p and g)
    evaluated = len(comparable)
    accuracy = (tp + tn) / evaluated if evaluated else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    impact_counts: Dict[str, int] = {}
    for r in results:
        effect = r.get("issue_effect", "unsupported")
        impact_counts[effect] = impact_counts.get(effect, 0) + 1

    return {
        "total_samples": total,
        "evaluated_samples": evaluated,
        "skipped_samples": skipped,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "issue_effect_counts": impact_counts,
    }


def save_json(path: Path, payload: Any):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_results_csv(path: Path, results: List[Dict[str, Any]]):
    fields = [
        "index",
        "original_label",
        "reeval_label",
        "issue_effect",
        "review_text",
        "reeval_explanation",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(results, start=1):
            writer.writerow(
                {
                    "index": idx,
                    "original_label": int(bool(row.get("original_label", False))),
                    "reeval_label": (
                        None
                        if row.get("reeval_label") is None
                        else int(bool(row.get("reeval_label")))
                    ),
                    "issue_effect": row.get("issue_effect"),
                    "review_text": row.get("review_text", ""),
                    "reeval_explanation": row.get("reeval_explanation", ""),
                    "error": row.get("error", ""),
                }
            )


def load_existing_results(logdir: Path) -> List[Dict[str, Any]] | None:
    path = logdir / "review_reeval_samples.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved reviews against the standard verifier rubric."
    )
    parser.add_argument("--logdir", required=True, help="directory containing verifier_samples.json")
    parser.add_argument("--concurrency", type=int, default=8, help="parallelism for LLM calls")
    parser.add_argument(
        "--force",
        action="store_true",
        help="force re-run even if review_reeval_samples.json already exists",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5-mini",
        help="base model to query for re-evaluation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("reeval")
    logdir = Path(args.logdir)

    logger.info("Loading verifier_samples from %s", logdir)
    samples = load_verifier_samples(logdir)
    logger.info("Loaded %d samples for re-evaluation", len(samples))

    existing = None if args.force else load_existing_results(logdir)
    if existing is not None:
        logger.info("Existing review re-evaluation found; recomputing statistics.")
        results = existing
    else:
        logger.info(
            "Running standard verifier rubric re-evaluations (model=%s, concurrency=%d)",
            args.model,
            args.concurrency,
        )
        results = asyncio.run(
            reevaluate_reviews(samples, model_name=args.model, concurrency=args.concurrency)
        )
        save_json(logdir / "review_reeval_samples.json", results)
        export_results_csv(logdir / "review_reeval_samples.csv", results)
        logger.info("Saved per-sample re-evaluations to %s", logdir)

    stats = compute_review_stats(results)
    save_json(logdir / "review_reeval_stats.json", stats)
    logger.info(
        "Review quality vs. standard verifier -- evaluated=%d, accuracy=%.4f, precision=%s, recall=%s, f1=%s",
        stats["evaluated_samples"],
        stats["accuracy"] if stats["accuracy"] is not None else float("nan"),
        stats["precision"],
        stats["recall"],
        stats["f1"],
    )
    logger.info("Issue effect counts: %s", stats.get("issue_effect_counts"))


if __name__ == "__main__":
    main()
