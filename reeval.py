import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import asyncio

from litellm import acompletion

# Explicit API configuration for litellm calls
# Set these at the top of this file; no CLI overrides.
API_BASE = "https://api.openai.com/v1"
API_KEY = ""

def extract_xml_content(text: str, tag: str) -> str | None:
    import re as _re
    flags = _re.DOTALL | 0
    pattern = rf"<{_re.escape(tag)}(?:\s+[^>]*)?\s*>(.*?)</\s*{_re.escape(tag)}\s*>"
    last_content = None
    for m in _re.finditer(pattern, text, flags):
        last_content = m.group(1)
    return last_content.strip() if last_content is not None else None


def load_verifier_samples(logdir: Path) -> List[Dict[str, Any]]:
    vs_path = logdir / "verifier_samples.json"
    if not vs_path.exists():
        raise FileNotFoundError(f"verifier_samples.json not found in {logdir}")
    with vs_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("verifier_samples.json must contain a list of samples")
    return samples


def filter_disagreements(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in samples if bool(s.get("pred_label", False)) != bool(s.get("gt_label", False))]


async def _compare_one(sample: Dict[str, Any], sem: asyncio.Semaphore) -> Dict[str, Any]:
    problem = sample.get("problem", "")
    proof = sample.get("proof", "")
    pred_text = sample.get("pred_text", "")
    gt_text = sample.get("gt_text", "")
    pred_label = bool(sample.get("pred_label", False))
    gt_label = bool(sample.get("gt_label", False))

    prompt = (
        "You are analyzing disagreement BETWEEN TWO REVIEWS, not re-judging the math solution.\n"
        "Given two review texts for the same solution with opposite labels, explain briefly why they disagree (focus on review behavior).\n"
        "Then classify the disagreement cause into EXACTLY ONE of:\n"
        "- missed_critical_error: one reviewer missed a decisive mistake or key point.\n"
        "- different_criterion: they used different acceptance criteria, rigor threshold or standards.\n"
        "- misinterpretation: one reviewer misunderstood the problem or the solution's intent.\n"
        "- other: none of the above.\n"
        "Additionally, ATTRIBUTE the cause to a specific reviewer:\n"
        "- Use <actor>A</actor> if Review A (pred_label/pred_text) is primarily responsible; <actor>B</actor> if Review B (gt_label/gt_text) is responsible.\n"
        "- Provide a concise behavioral tag in <behavior>...</behavior> choosing ONE from: A_more_strict, B_more_strict, A_more_lenient, B_more_lenient, A_missed_error, B_missed_error, A_misinterpreted, B_misinterpreted, none.\n"
        "The <behavior> MUST be consistent with <class> and the explanation.\n"
        "Respond ONLY with: <explanation>...</explanation><class>...</class><actor>...</actor><behavior>...</behavior>.\n\n"
        f"<problem>{problem}</problem>\n"
        f"<solution>{proof}</solution>\n\n"
        f"<review_a label=\"{pred_label}\">{pred_text}</review_a>\n"
        f"<review_b label=\"{gt_label}\">{gt_text}</review_b>\n"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant for analyzing disagreements between mathematical reviews."},
        {"role": "user", "content": prompt},
    ]

    try:
        async with sem:
            resp = await acompletion(
                model="openai/gpt-5-mini",
                messages=messages,
                api_base=API_BASE,
                api_key=API_KEY,
                drop_params=True,
            )
        content = resp.choices[0].message["content"] if resp is not None else ""
    except Exception as e:
        content = f"Error: {e}"

    explanation = extract_xml_content(content, "explanation") or ""
    classification = (extract_xml_content(content, "class") or "").strip()
    actor = (extract_xml_content(content, "actor") or "").strip()
    behavior = (extract_xml_content(content, "behavior") or "").strip()

    return {
        "problem": problem,
        "proof": proof,
        "pred_label": pred_label,
        "gt_label": gt_label,
        "pred_text": pred_text,
        "gt_text": gt_text,
        "analysis": content,
        "explanation": explanation,
        "classification": classification,
        "actor": actor,
        "behavior": behavior,
    }


async def analyze_disagreements_async(samples: List[Dict[str, Any]], concurrency: int = 8) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(_compare_one(s, sem)) for s in samples]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out.append({"error": str(r), **samples[i]})
        else:
            out.append(r)
    return out


def save_outputs(logdir: Path, disagreements: List[Dict[str, Any]]):
    out_path = logdir / "verifier_disagreements.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(disagreements, f, ensure_ascii=False, indent=2)

def export_disagreements_csv(logdir: Path, disagreements: List[Dict[str, Any]]) -> Path:
    """Export disagreements to CSV for easy review."""
    csv_path = logdir / "verifier_disagreements.csv"
    fields = [
        "index",
        "classification",
        "actor",
        "behavior",
        "pred_label",
        "gt_label",
        "problem",
        "explanation",
        "pred_text",
        "gt_text",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, s in enumerate(disagreements, start=1):
            row = {
                "index": i,
                "classification": (s.get("classification") or "").strip(),
                "actor": (s.get("actor") or "").strip(),
                "behavior": (s.get("behavior") or "").strip(),
                "pred_label": int(bool(s.get("pred_label", False))),
                "gt_label": int(bool(s.get("gt_label", False))),
                "problem": s.get("problem", ""),
                "explanation": s.get("explanation", ""),
                "pred_text": s.get("pred_text", ""),
                "gt_text": s.get("gt_text", ""),
            }
            writer.writerow(row)
    return csv_path


def log_classification_counts(disagreements: List[Dict[str, Any]], logger: logging.Logger):
    by_class: Dict[str, int] = {}
    by_actor: Dict[str, int] = {}
    by_class_actor: Dict[str, int] = {}
    by_behavior: Dict[str, int] = {}
    for s in disagreements:
        cls = (s.get("classification") or "").strip() or "unknown"
        act = (s.get("actor") or "").strip() or "unknown"
        beh = (s.get("behavior") or "").strip() or "none"
        by_class[cls] = by_class.get(cls, 0) + 1
        by_actor[act] = by_actor.get(act, 0) + 1
        key = f"{cls}:{act}"
        by_class_actor[key] = by_class_actor.get(key, 0) + 1
        by_behavior[beh] = by_behavior.get(beh, 0) + 1

    total = len(disagreements)
    logger.info("Disagreement counts (total=%d)", total)
    logger.info("  By class:")
    for k, v in sorted(by_class.items(), key=lambda x: (-x[1], x[0])):
        logger.info("    %s: %d", k, v)
    logger.info("  By actor:")
    for k, v in sorted(by_actor.items(), key=lambda x: (-x[1], x[0])):
        logger.info("    %s: %d", k, v)
    logger.info("  By class+actor:")
    for k, v in sorted(by_class_actor.items(), key=lambda x: (-x[1], x[0])):
        logger.info("    %s: %d", k, v)
    logger.info("  By behavior:")
    for k, v in sorted(by_behavior.items(), key=lambda x: (-x[1], x[0])):
        logger.info("    %s: %d", k, v)


def load_existing_disagreements(logdir: Path) -> List[Dict[str, Any]] | None:
    p = logdir / "verifier_disagreements.json"
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate and classify disagreements in verifier_samples")
    parser.add_argument("--logdir", required=True, help="path to the log directory containing verifier_samples.json")
    parser.add_argument("--concurrency", type=int, default=8, help="parallelism for LLM calls")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("reeval")

    logdir = Path(args.logdir)
    logger.info("Loading verifier_samples from %s", logdir)
    samples = load_verifier_samples(logdir)
    disagreements = filter_disagreements(samples)
    logger.info("Found %d disagreements to analyze", len(disagreements))

    if not disagreements:
        logger.info("No disagreements detected; nothing to analyze.")
        return

    # If analysis already exists, do not re-call the LLM; just reload and print stats
    existing = load_existing_disagreements(logdir)
    if existing is not None:
        logger.info("Existing disagreement analysis found; reloading and printing statistics.")
        log_classification_counts(existing, logger)
        csv_path = export_disagreements_csv(logdir, existing)
        logger.info("Exported disagreement CSV to %s", csv_path)
        return

    logger.info("Running gpt-5-mini analyses in parallel (concurrency=%d)", args.concurrency)
    analyzed = asyncio.run(analyze_disagreements_async(disagreements, concurrency=args.concurrency))
    save_outputs(logdir, analyzed)
    log_classification_counts(analyzed, logger)
    csv_path = export_disagreements_csv(logdir, analyzed)
    logger.info("Exported disagreement CSV to %s", csv_path)
    logger.info("Saved disagreement analyses to %s", logdir / "verifier_disagreements.json")


if __name__ == "__main__":
    main()
