import asyncio
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), ".env"))

from eval_lib import (
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    AnswerPrecisionMetric,
    RestrictedRefusalMetric,
)

FOLDER = os.path.dirname(os.path.abspath(__file__))
ANSWERS = os.path.join(FOLDER, "data", "homework_rag_answers.json")
RESULTS = os.path.join(FOLDER, "data", "homework_results_evaliqa.json")

MODEL = "gpt-4o-mini"
THRESHOLD = 0.7

RUSSIAN_REFUSALS = [
    "не могу найти", "нет информации", "отсутствует информация",
    "не содержится", "не указано", "в документации нет",
    "не нашел", "не нашёл", "не располагаю", "нет данных",
    "информация отсутствует", "не предоставлена", "не могу ответить",
    "не упоминается", "не указан", "не содержит",
]


def build_cases(rows):
    cases = []
    for row in rows:
        if row.get("error"):
            continue
        cases.append(EvalTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["expected_response"],
            retrieval_context=row["contexts"],
        ))
    return cases


def flatten(result):
    cases = []
    for entry in result:
        if isinstance(entry, tuple):
            cases.extend(entry[1])
        else:
            cases.append(entry)
    return cases


async def run_block(name, cases, metrics):
    print("")
    print("=" * 70)
    print(f"{name}: {len(cases)} cases")
    print("=" * 70)
    raw = await evaluate(test_cases=cases, metrics=metrics, verbose=False)
    return flatten(raw)


def report(name, cases):
    scores, cost, details = {}, 0.0, []
    for case in cases:
        row = {"question": case.input, "metrics": {}}
        for metric in case.metrics_data:
            if metric.score is not None:
                scores.setdefault(metric.name, []).append(metric.score)
            cost += metric.evaluation_cost or 0.0
            row["metrics"][metric.name] = {
                "score": metric.score,
                "success": metric.success,
                "reason": (metric.reason or "")[:400],
            }
        details.append(row)

    means = {k: sum(v) / len(v) for k, v in scores.items()}
    below = {k: sum(1 for x in v if x < THRESHOLD) for k, v in scores.items()}

    print(f"\n{name} — {len(cases)} cases, cost ${cost:.4f}")
    for metric, value in means.items():
        print(f"  {metric:<30} {value:.3f}   below {THRESHOLD}: {below[metric]}/{len(scores[metric])}")

    return {"means": means, "below_threshold": below, "cost": cost,
            "cases": len(cases), "details": details}


async def main(limit=None):
    data = json.load(open(ANSWERS, encoding="utf-8"))
    output = {}

    happy = build_cases(data["happy_path"])[:limit]
    output["happy_path"] = report("HAPPY PATH", await run_block("HAPPY PATH", happy, [
        AnswerRelevancyMetric(model=MODEL, threshold=THRESHOLD),
        FaithfulnessMetric(model=MODEL, threshold=THRESHOLD),
        ContextualRelevancyMetric(model=MODEL, threshold=THRESHOLD),
    ]))

    negative = build_cases(data["negative"])[:limit]
    output["negative"] = report("NEGATIVE", await run_block("NEGATIVE", negative, [
        AnswerPrecisionMetric(model=MODEL, threshold=THRESHOLD),
        RestrictedRefusalMetric(custom_patterns=RUSSIAN_REFUSALS),
    ]))

    metamorphic = build_cases(data["metamorphic"])[:limit]
    output["metamorphic"] = report("METAMORPHIC", await run_block("METAMORPHIC", metamorphic, [
        AnswerPrecisionMetric(model=MODEL, threshold=THRESHOLD),
    ]))

    total = sum(block["cost"] for block in output.values())
    print("")
    print("=" * 70)
    print(f"TOTAL COST: ${total:.4f}")
    print("=" * 70)

    json.dump(output, open(RESULTS, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(f"Saved to {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
