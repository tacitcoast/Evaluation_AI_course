import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from ai_client import openai_chat_v3, google_chat_v3
from benchmarks import (
    run_mmlu_pro,
    run_truthfulqa,
    run_hellaswag,
    run_bigbench_hard,
    run_mathqa,
    run_boolq,
    run_lambada,
    run_winogrande,
    run_ifeval,
    run_bbq,
)

NUM_SAMPLES = 10
TEMPERATURE = 0.1
RESULTS_FILE = "benchmark_results.json"

MMLU_CATEGORY = "psychology"
BBH_TASK = "causal_judgement"
BBQ_CATEGORY = "sexual_orientation"

MODELS = {
    "gpt-5.4-nano": lambda p: openai_chat_v3(p, "gpt-5.4-nano", TEMPERATURE),
    "gemini-3-flash-preview": lambda p: google_chat_v3(p, "gemini-3-flash-preview", TEMPERATURE),
}

BENCHMARKS = [
    (f"MMLU-Pro ({MMLU_CATEGORY})", lambda chat: run_mmlu_pro(MMLU_CATEGORY, NUM_SAMPLES, chat)),
    ("TruthfulQA", lambda chat: run_truthfulqa(NUM_SAMPLES, chat)),
    ("HellaSwag", lambda chat: run_hellaswag(NUM_SAMPLES, chat)),
    (f"BIG-Bench Hard ({BBH_TASK})", lambda chat: run_bigbench_hard(BBH_TASK, NUM_SAMPLES, chat)),
    ("MathQA", lambda chat: run_mathqa(NUM_SAMPLES, chat)),
    ("BoolQ", lambda chat: run_boolq(NUM_SAMPLES, chat)),
    ("LAMBADA", lambda chat: run_lambada(NUM_SAMPLES, chat)),
    ("Winogrande", lambda chat: run_winogrande(NUM_SAMPLES, chat)),
    ("IFEval", lambda chat: run_ifeval(NUM_SAMPLES, chat)),
    (f"BBQ ({BBQ_CATEGORY})", lambda chat: run_bbq(BBQ_CATEGORY, NUM_SAMPLES, chat)),
]


def load_done():
    if os.path.exists(RESULTS_FILE):
        return json.load(open(RESULTS_FILE, encoding="utf-8"))
    return {}


def save(results):
    json.dump(results, open(RESULTS_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def main():
    print("=" * 70)
    print("BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Questions per benchmark: {NUM_SAMPLES}, temperature: {TEMPERATURE}")
    print(f"Benchmarks: {len(BENCHMARKS)} per model")

    results = load_done()

    for model_name, chat in MODELS.items():
        results.setdefault(model_name, {})
        for bench_name, run in BENCHMARKS:
            if bench_name in results[model_name]:
                print(f"\nSKIP {model_name} / {bench_name} - already in {RESULTS_FILE}")
                continue

            print("")
            print("#" * 70)
            print(f"# {model_name} / {bench_name}")
            print("#" * 70)

            start = time.time()
            try:
                outcome = run(chat)
                outcome["seconds"] = round(time.time() - start, 1)
                outcome["error"] = None
            except Exception as exc:
                outcome = {"correct": None, "total": NUM_SAMPLES, "accuracy": None,
                           "seconds": round(time.time() - start, 1),
                           "error": f"{type(exc).__name__}: {exc}"}
                print(f"FAILED: {outcome['error']}")

            results[model_name][bench_name] = outcome
            save(results)

    rows = []
    for bench_name, _ in BENCHMARKS:
        row = {"Benchmark": bench_name}
        for model_name in MODELS:
            data = results.get(model_name, {}).get(bench_name, {})
            accuracy = data.get("accuracy")
            row[model_name] = "n/a" if accuracy is None else f"{accuracy:.2f}"
        rows.append(row)

    print("")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(pd.DataFrame(rows).to_string(index=False))

    print("")
    failures = [(m, b, d["error"]) for m, benches in results.items()
                for b, d in benches.items() if d.get("error")]
    if failures:
        print("FAILED RUNS:")
        for model_name, bench_name, error in failures:
            print(f"  {model_name} / {bench_name}: {error[:90]}")
    else:
        print("No failed runs")

    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
