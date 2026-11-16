import asyncio
import json
import os
import re

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), ".env"))

from eval_lib.metrics.faithfulness_metric import faithfulness as fth

fth._TOKEN = re.compile(r"[a-zа-яё0-9]{3,}")

from eval_lib import evaluate, EvalTestCase, FaithfulnessMetric

FOLDER = os.path.dirname(os.path.abspath(__file__))
ANSWERS = os.path.join(FOLDER, "data", "homework_rag_answers.json")
RESULTS = os.path.join(FOLDER, "data", "homework_faithfulness_fixed.json")

MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7


async def main():
    data = json.load(open(ANSWERS, encoding="utf-8"))
    rows = [r for r in data["happy_path"] if not r.get("error")]

    cases = [EvalTestCase(
        input=r["question"],
        actual_output=r["answer"],
        expected_output=r["expected_response"],
        retrieval_context=r["contexts"],
    ) for r in rows]

    raw = await evaluate(test_cases=cases,
                         metrics=[FaithfulnessMetric(model=MODEL, threshold=THRESHOLD)],
                         verbose=False)

    results = []
    for entry in raw:
        results.extend(entry[1] if isinstance(entry, tuple) else [entry])

    scores, cost, details = [], 0.0, []
    for case in results:
        metric = case.metrics_data[0]
        scores.append(metric.score)
        cost += metric.evaluation_cost or 0.0
        details.append({"question": case.input, "score": metric.score,
                        "reason": (metric.reason or "")[:300]})

    mean = sum(scores) / len(scores)
    zeros = sum(1 for s in scores if s == 0.0)
    below = sum(1 for s in scores if s < THRESHOLD)

    print("=" * 70)
    print("FAITHFULNESS WITH CYRILLIC-AWARE TOKEN REGEX")
    print("=" * 70)
    print(f"  cases:        {len(scores)}")
    print(f"  mean score:   {mean:.3f}")
    print(f"  zeros:        {zeros}")
    print(f"  below {THRESHOLD}:   {below}")
    print(f"  cost:         ${cost:.4f}")

    json.dump({"mean": mean, "zeros": zeros, "below_threshold": below,
               "cost": cost, "details": details},
              open(RESULTS, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved to {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())
