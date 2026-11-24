# homework_traces_evaliqa.py
"""Анализ трейсов в EvaliQA: метрики, которые читают не ответ, а траекторию.

EvaliQA (eval-ai-library) держит трейс в EvalTestCase.execution_trace как список
TraceStep с полями name, input, output, status, error, duration_ms. Читают его
три метрики:
  PlanningQuality  — похожа ли траектория на осмысленный план нужной длины
  LoopDetection    — нет ли повторов и циклов в цепочке шагов
  ContextDecay     — не теряются ли по ходу шагов факты из задачи

Скрипт работает в двух режимах.

  shallow — трейс, который даёт боевой агент. /ask возвращает только список имён
            инструментов, поэтому у шагов нет ни аргументов, ни результатов.
            Это и есть трейс системы под тестом, как он доступен снаружи.

  deep    — трейс локального прогона агента (homework_trace_local.py), где у
            каждого шага есть аргументы вызова, результат и статус.

Сравнение двух режимов на одних и тех же вопросах показывает, что именно
теряется из-за того, что сервис не отдаёт аргументы и результаты вызовов.

Запуск: cd Lecture5.Agents && ./.venv/bin/python homework_traces_evaliqa.py
"""

import asyncio
import contextlib
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval_lib import (                                      # noqa: E402
    EvalTestCase,
    TraceStep,
    PlanningQualityMetric,
    LoopDetectionMetric,
    ContextDecayMetric,
)

RESPONSES = Path("data/agent_responses.json")
LOCAL_TRACES = Path("data/traces_local.json")
OUT_SHALLOW = Path("data/scores_traces_shallow.json")
OUT_DEEP = Path("data/scores_traces_deep.json")

SLICES = ["happy_path", "negative", "metamorphic"]


def shallow_trace(record) -> list:
    """Трейс из того, что отдаёт /ask: только имена инструментов по порядку.

    Аргументы и результаты сервис не возвращает, поэтому input и output
    остаются пустыми. Статус шага неизвестен: из ответа не видно, сработал
    инструмент или вернул пустоту.
    """
    steps = []
    for i, name in enumerate(record.get("tools_used") or [], 1):
        steps.append(TraceStep(
            step_id=str(i),
            type="tool",
            name=name,
            input=None,
            output=None,
            status="unknown",
        ))
    if record.get("error"):
        steps.append(TraceStep(
            step_id=str(len(steps) + 1),
            type="error",
            name="http_response",
            status="error",
            error=str(record["error"])[:200],
            error_type="HTTP 500",
            duration_ms=float(record.get("latency_sec") or 0) * 1000,
        ))
    return steps


def deep_trace(record) -> list:
    """Трейс локального прогона: шаги уже записаны со всеми полями."""
    return [TraceStep(**step) for step in record.get("trace", [])]


def item_count(value):
    """Сколько объявлений в значении: трейс сворачивает длинные списки."""
    if isinstance(value, dict) and "items" in value:
        return value["items"]
    if isinstance(value, list):
        return len(value)
    return None


def chain_integrity(record):
    """Доживают ли объявления от шага к шагу.

    Инструменты принимают список объявлений аргументом, и агент переотправляет
    его через модель на каждом шаге. Сравниваем, сколько объявлений вышло из
    предыдущего шага и сколько пришло в следующий.
    """
    links = []
    steps = record.get("trace", [])
    for prev, step in zip(steps, steps[1:]):
        produced = (prev.get("metadata") or {}).get("items")
        if produced is None:
            produced = item_count(prev.get("output"))
        received = None
        for key in ("offers", "prices"):
            if isinstance(step.get("input"), dict) and key in step["input"]:
                received = item_count(step["input"][key])
        if produced is None or received is None:
            continue
        links.append({
            "from": prev["name"],
            "to": step["name"],
            "produced": produced,
            "received": received,
            "lost": produced - received,
        })
    return links


def build_context_facts(record):
    """Факты из задачи и шаг, на котором инструмент их действительно применил.

    ContextDecay принимает факты с introduced_at, used_at и correctly_used.
    Берём то, что пользователь задал в вопросе: ссылку, границу цены, целевую
    валюту, и отдельно — объявления, которые должны дожить до следующего шага.
    """
    facts = []
    steps = record.get("trace", [])
    question = record["question"]

    for i, step in enumerate(steps):
        args = step.get("input") or {}
        if not isinstance(args, dict):
            continue

        if "url" in args:
            wanted = (record.get("urls") or [None])[0]
            facts.append({"fact": "ссылка из задачи", "introduced_at": 0, "used_at": i,
                          "correctly_used": bool(wanted and str(args["url"]) == wanted)})

        for key in ("max_price", "min_price"):
            if args.get(key) not in (None, "None"):
                numbers = {re.sub(r"\D", "", n) for n in re.findall(r"\d[\d\s]*", question)}
                facts.append({"fact": f"{key} из задачи", "introduced_at": 0, "used_at": i,
                              "correctly_used": str(args[key]) in numbers})

        if "target_currency" in args:
            cur = str(args["target_currency"]).upper()
            asked = ("RUB" if "рубл" in question.lower() else
                     "USD" if "доллар" in question.lower() else
                     "EUR" if "евро" in question.lower() else None)
            facts.append({"fact": "целевая валюта из задачи", "introduced_at": 0, "used_at": i,
                          "correctly_used": asked is not None and cur == asked})

    for n, link in enumerate(chain_integrity(record), 1):
        facts.append({
            "fact": f"объявления из {link['from']} в {link['to']}",
            "introduced_at": n - 1,
            "used_at": n,
            "correctly_used": link["lost"] == 0,
        })

    return facts


def build_case(record, steps) -> EvalTestCase:
    output = record.get("output") or ""
    if record.get("error"):
        output = f"[сбой сервиса] {record['error']}"
    task = record["question"]
    if record.get("urls"):
        task += "\nСсылки: " + ", ".join(record["urls"])
    return EvalTestCase(
        input=task,
        actual_output=output,
        expected_output=record.get("expected_response") or "",
        tools_called=list(record.get("tools_used") or []),
        expected_tools=list(record.get("expected_tools") or []),
        execution_trace=steps,
        name=record["id"],
    )


async def score_one(metric, case, name):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = await metric.evaluate(case)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        score = result.get("score") if isinstance(result, dict) else getattr(result, "score", None)
        reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
        return {"metric": name, "score": score, "reason": reason,
                "success": score is not None and score >= metric.threshold, "error": None}
    except Exception as exc:
        return {"metric": name, "score": None, "reason": None, "success": False,
                "error": f"{type(exc).__name__}: {exc}"}


async def run(records, trace_fn, out_path: Path, label: str):
    print("=" * 72)
    print(f"Анализ трейсов в EvaliQA: {label}")
    print("=" * 72)
    print(f"Кейсов: {len(records)}\n")

    results = []
    for i, record in enumerate(records, 1):
        steps = trace_fn(record)
        case = build_case(record, steps)

        # PlanningQuality сравнивает длину траектории с ожидаемой: сколько
        # инструментов задача требует по датасету
        expected = len(record.get("expected_tools") or []) or None
        # ContextDecay без context_facts возвращает 1.0 и ничего не измеряет,
        # поэтому факты собираем из трейса там, где он их содержит
        facts = build_context_facts(record) if record.get("trace") else []
        metrics = [
            ("planning_quality", PlanningQualityMetric(expected_tool_count=expected)),
            ("loop_detection", LoopDetectionMetric()),
            ("context_decay", ContextDecayMetric(context_facts=facts or None)),
        ]

        row = {
            "id": record["id"],
            "test_type": record.get("test_type", ""),
            "question": record["question"],
            "trace_steps": len(steps),
            "steps_with_arguments": sum(1 for s in steps if s.input is not None),
            "steps_with_result": sum(1 for s in steps if s.output is not None),
            "agent_error": record.get("error"),
            "context_facts": facts,
            "chain": chain_integrity(record),
            "scores": {},
        }

        line = [f"[{i}/{len(records)}] {record['id']} шагов {len(steps)}"]
        for name, metric in metrics:
            scored = await score_one(metric, case, name)
            row["scores"][name] = scored
            value = "—" if scored["score"] is None else f"{scored['score']:.2f}"
            line.append(f"{name.split('_')[0]}={value}")
        print(" ".join(line))

        results.append(row)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summarize(results, label)
    print(f"\nПодробности: {out_path}")
    return results


def summarize(results, label):
    names = ["planning_quality", "loop_detection", "context_decay"]

    def avg(rows, name):
        values = [r["scores"][name]["score"] for r in rows
                  if r["scores"].get(name, {}).get("score") is not None]
        return sum(values) / len(values) if values else None

    print("\n" + "-" * 72)
    print(f"Средние по трейсам ({label}):")
    print(f"{'срез':<14}{'n':>4}  " + "  ".join(f"{n.split('_')[0]:>9}" for n in names))

    def line(lbl, rows):
        if not rows:
            return
        cells = []
        for name in names:
            value = avg(rows, name)
            cells.append("     —   " if value is None else f"{value:>9.3f}")
        print(f"{lbl:<14}{len(rows):>4}  " + "  ".join(cells))

    line("все", results)
    for test_type in SLICES:
        line(test_type, [r for r in results if r["test_type"] == test_type])

    links = [l for r in results for l in (r.get("chain") or [])]
    if links:
        lossy = [l for l in links if l["lost"] > 0]
        print(f"\nПереходов между шагами: {len(links)}, с потерей объявлений: {len(lossy)}")
        for l in lossy:
            print(f"  {l['from']} -> {l['to']}: вышло {l['produced']}, "
                  f"пришло {l['received']}, потеряно {l['lost']}")

    total_steps = sum(r["trace_steps"] for r in results)
    with_args = sum(r["steps_with_arguments"] for r in results)
    with_result = sum(r["steps_with_result"] for r in results)
    print(f"\nШагов в трейсах всего: {total_steps}")
    print(f"  с аргументами вызова: {with_args}")
    print(f"  с результатом вызова: {with_result}")

    broken = [(r["id"], n, s["error"]) for r in results
              for n, s in r["scores"].items() if s["error"]]
    if broken:
        print(f"\nМетрика не посчиталась в {len(broken)} случаях:")
        for rid, name, err in broken[:8]:
            print(f"  [{rid}] {name}: {err[:70]}")


def main():
    if not RESPONSES.exists():
        raise SystemExit(f"Нет {RESPONSES}. Сначала homework_collect.py")

    records = json.loads(RESPONSES.read_text(encoding="utf-8"))
    asyncio.run(run(records, shallow_trace, OUT_SHALLOW,
                    "трейс боевого агента, только имена инструментов"))

    if LOCAL_TRACES.exists():
        local = json.loads(LOCAL_TRACES.read_text(encoding="utf-8"))
        print("\n")
        asyncio.run(run(local, deep_trace, OUT_DEEP,
                        "трейс локального прогона, с аргументами и результатами"))
    else:
        print(f"\nНет {LOCAL_TRACES}: глубокие трейсы не считаю. "
              "Сначала homework_trace_local.py")


if __name__ == "__main__":
    main()
