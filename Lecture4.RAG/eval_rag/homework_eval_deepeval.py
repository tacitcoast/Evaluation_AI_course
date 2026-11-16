import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), ".env"))

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "YES")

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)

FOLDER = os.path.dirname(os.path.abspath(__file__))
ANSWERS = os.path.join(FOLDER, "data", "homework_rag_answers.json")
RESULTS = os.path.join(FOLDER, "data", "homework_results_deepeval.json")

MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7


def build_metrics():
    return [
        AnswerRelevancyMetric(model=MODEL, threshold=THRESHOLD, async_mode=False, include_reason=True),
        FaithfulnessMetric(model=MODEL, threshold=THRESHOLD, async_mode=False, include_reason=True),
        ContextualRelevancyMetric(model=MODEL, threshold=THRESHOLD, async_mode=False, include_reason=True),
    ]


def main(limit=None):
    data = json.load(open(ANSWERS, encoding="utf-8"))
    rows = [r for r in data["happy_path"] if not r.get("error")][:limit]

    print("=" * 70)
    print(f"DEEPEVAL — HAPPY PATH: {len(rows)} cases, judge {MODEL}")
    print("=" * 70)

    collected = []
    if os.path.exists(RESULTS):
        collected = json.load(open(RESULTS, encoding="utf-8"))
    done = {item["question"] for item in collected}

    metrics = build_metrics()
    start = time.time()

    for i, row in enumerate(rows, 1):
        if row["question"] in done:
            continue

        case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["expected_response"],
            retrieval_context=row["contexts"],
        )

        item = {"question": row["question"], "test_type": row.get("test_type"), "scores": {}}
        for metric in metrics:
            try:
                metric.measure(case)
                item["scores"][metric.__class__.__name__] = {
                    "score": metric.score,
                    "reason": metric.reason,
                }
            except Exception as exc:
                item["scores"][metric.__class__.__name__] = {
                    "score": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }

        collected.append(item)
        json.dump(collected, open(RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        line = "  ".join(
            f"{name[:12]}={v['score']:.2f}" if v["score"] is not None else f"{name[:12]}=ERR"
            for name, v in item["scores"].items()
        )
        print(f"[{i}/{len(rows)}] {line}  | {row['question'][:45]}", flush=True)

    print("")
    print("=" * 70)
    print("AVERAGE SCORES")
    print("=" * 70)
    means = {}
    for item in collected:
        for name, value in item["scores"].items():
            if value["score"] is not None:
                means.setdefault(name, []).append(value["score"])

    for name, values in means.items():
        below = sum(1 for v in values if v < THRESHOLD)
        print(f"  {name:<28} {sum(values) / len(values):.3f}   below {THRESHOLD}: {below}/{len(values)}")

    print("")
    print(f"Time: {time.time() - start:.0f}s, saved to {RESULTS}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
