# homework_evaluate_deepeval.py
"""Стадия 2a: оценка собранных ответов агента фреймворком DeepEval.

Читает data/agent_responses.json (стадия 1, homework_collect.py) и считает три
метрики: AnswerRelevancy, ToolCorrectness, TaskCompletion.

Метрики считаются по одному кейсу за раз через measure(), а не одним вызовом
evaluate(), по двум причинам: падение судьи на одном кейсе не роняет прогон,
и рядом с каждой оценкой сохраняется id строки датасета, её тип и объяснение
судьи. Агрегат без возможности провалиться в конкретные примеры бесполезен.

Запуск: cd Lecture5.Agents && ./.venv/bin/python homework_evaluate_deepeval.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "deepeval_lib"))

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("CONFIDENT_AI_OPT_OUT", "YES")

import homework_config as cfg                                   # noqa: E402
from deepeval_custom_llm import create_proxy_model              # noqa: E402
from deepeval.metrics import (                                  # noqa: E402
    AnswerRelevancyMetric,
    ToolCorrectnessMetric,
    TaskCompletionMetric,
)
from deepeval.test_case import LLMTestCase, ToolCall            # noqa: E402

RESPONSES = Path("data/agent_responses.json")
OUT = Path("data/scores_deepeval.json")
THRESHOLD = 0.7


def to_tool_calls(names):
    return [ToolCall(name=n) for n in names]


def build_case(record) -> LLMTestCase:
    """Собирает тест-кейс. Ошибка сервиса остаётся в actual_output как есть:
    пользователь увидел бы именно её, и судья должен оценивать её, а не пустоту."""
    output = record.get("output") or ""
    if record.get("error"):
        output = f"[сбой сервиса] {record['error']}"

    task = record["question"]
    if record.get("urls"):
        task += "\nСсылки: " + ", ".join(record["urls"])

    return LLMTestCase(
        input=task,
        actual_output=output,
        expected_output=record.get("expected_response") or None,
        tools_called=to_tool_calls(record.get("tools_used") or []),
        expected_tools=to_tool_calls(record.get("expected_tools") or []),
    )


def score_one(metric, case, name):
    """Считает одну метрику по одному кейсу, не роняя прогон на исключении."""
    try:
        metric.measure(case)
        return {
            "metric": name,
            "score": metric.score,
            "success": bool(metric.success),
            "reason": getattr(metric, "reason", None),
            "cost": getattr(metric, "evaluation_cost", None),
            "error": None,
        }
    except Exception as exc:
        return {
            "metric": name,
            "score": None,
            "success": False,
            "reason": None,
            "cost": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run(responses_path: Path = RESPONSES, out_path: Path = OUT,
        threshold: float = THRESHOLD, limit=None):
    if not responses_path.exists():
        raise SystemExit(
            f"Нет файла {responses_path}. Сначала запустите homework_collect.py")

    records = json.loads(responses_path.read_text(encoding="utf-8"))
    if limit is not None:
        records = records[:limit]
    print(f"Кейсов к оценке: {len(records)}")

    judge = create_proxy_model(
        model=cfg.JUDGE_MODEL,
        api_key=cfg.require("PROXY_API_KEY"),
        base_url=cfg.PROXY_BASE_URL,
    )
    print(f"Судья: {cfg.JUDGE_MODEL} через {cfg.PROXY_BASE_URL}\n")

    # ToolCorrectness сравнивает списки инструментов и в LLM не ходит, но в
    # deepeval 4.0.7 всё равно требует модель в конструкторе: без неё лезет за
    # OPENAI_API_KEY напрямую в OpenAI, минуя прокси. Поэтому судья передаётся.
    # should_exact_match=False, потому что агент законно может вызвать
    # extract_offers дважды на двух ссылках; важен состав, а не кратность.
    metrics = [
        ("answer_relevancy", lambda: AnswerRelevancyMetric(
            threshold=threshold, model=judge, include_reason=True)),
        ("tool_correctness", lambda: ToolCorrectnessMetric(
            threshold=threshold, model=judge, should_exact_match=False,
            should_consider_ordering=False)),
        ("task_completion", lambda: TaskCompletionMetric(
            threshold=threshold, model=judge, include_reason=True)),
    ]

    results = []
    for i, record in enumerate(records, 1):
        case = build_case(record)
        row = {
            "id": record["id"],
            "test_type": record.get("test_type", ""),
            "metamorphic_group": record.get("metamorphic_group", ""),
            "question": record["question"],
            "urls": record.get("urls", []),
            "tools_used": record.get("tools_used", []),
            "expected_tools": record.get("expected_tools", []),
            "agent_error": record.get("error"),
            "latency_sec": record.get("latency_sec"),
            "scores": {},
        }

        line = [f"[{i}/{len(records)}] {record['id']} {record.get('test_type', ''):<11}"]
        for name, make in metrics:
            scored = score_one(make(), case, name)
            row["scores"][name] = scored
            value = "—" if scored["score"] is None else f"{scored['score']:.2f}"
            line.append(f"{name.split('_')[0]}={value}")
        print(" ".join(line))

        results.append(row)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summarize(results, threshold)
    print(f"\nПодробности по каждому кейсу: {out_path}")
    return results


def summarize(results, threshold=THRESHOLD):
    print("\n" + "=" * 62)
    print("DeepEval: средние оценки")
    print("=" * 62)

    names = ["answer_relevancy", "tool_correctness", "task_completion"]
    print(f"{'срез':<14}{'n':>4}  " + "  ".join(f"{n.split('_')[0]:>9}" for n in names))

    def avg(rows, name):
        values = [r["scores"][name]["score"] for r in rows
                  if r["scores"].get(name, {}).get("score") is not None]
        return sum(values) / len(values) if values else None

    def line(label, rows):
        if not rows:
            return
        cells = []
        for name in names:
            value = avg(rows, name)
            cells.append("     —   " if value is None else f"{value:>9.3f}")
        print(f"{label:<14}{len(rows):>4}  " + "  ".join(cells))

    line("весь датасет", results)
    for test_type in ["happy_path", "negative", "metamorphic"]:
        line(test_type, [r for r in results if r["test_type"] == test_type])

    print(f"\nДоля кейсов выше порога {threshold}:")
    for name in names:
        rows = [r for r in results if r["scores"].get(name, {}).get("score") is not None]
        if not rows:
            continue
        passed = sum(1 for r in rows if r["scores"][name]["success"])
        print(f"  {name:<18} {passed} из {len(rows)}")

    failed = [r for r in results if r.get("agent_error")]
    print(f"\nЗапросов, на которых агент ответил ошибкой: {len(failed)}")

    cost = sum(
        s["cost"] or 0 for r in results for s in r["scores"].values())
    if cost:
        print(f"Стоимость работы судьи: ${cost:.4f}")

    broken = [(r["id"], n, s["error"]) for r in results
              for n, s in r["scores"].items() if s["error"]]
    if broken:
        print(f"\nМетрика не посчиталась в {len(broken)} случаях:")
        for rid, name, err in broken[:10]:
            print(f"  [{rid}] {name}: {err[:70]}")


if __name__ == "__main__":
    run(responses_path=RESPONSES, out_path=OUT, threshold=THRESHOLD, limit=None)
