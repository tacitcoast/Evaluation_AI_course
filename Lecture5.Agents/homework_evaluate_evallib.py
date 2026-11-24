# homework_evaluate_evallib.py
"""Стадия 2b: оценка тех же собранных ответов вторым фреймворком, eval-ai-library.

Читает тот же data/agent_responses.json, что и прогон DeepEval, поэтому оба
фреймворка судят буквально одни и те же ответы агента: расхождение в оценках
говорит о фреймворках, а не о том, что агент в двух прогонах ответил по-разному.

Метрики:
  AnswerRelevancy    — на всём датасете
  ToolCorrectness    — на всём датасете, состав инструментов без учёта порядка
  TaskSuccessRate    — на всём датасете
  ToolsError         — на всём датасете: ловит сбои инструментов в ответе
  RestrictedRefusal  — только на негативных кейсах: там отказ и есть верное
                       поведение, на happy path он был бы дефектом

Запуск: cd Lecture5.Agents && ./.venv/bin/python homework_evaluate_evallib.py
"""

import asyncio
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "eval_ai_library"))

import homework_config as cfg                                   # noqa: E402
from custom_proxy_llm import create_proxy_llm                   # noqa: E402
from eval_lib import (                                          # noqa: E402
    EvalTestCase,
    ToolCall,
    AnswerRelevancyMetric,
    ToolCorrectnessMetric,
    TaskSuccessRateMetric,
    ToolsErrorMetric,
    RestrictedRefusalMetric,
)

RESPONSES = Path("data/agent_responses.json")
OUT = Path("data/scores_evallib.json")
THRESHOLD = 0.7


def build_case(record) -> EvalTestCase:
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
        # eval_lib ждёт здесь именно строки, в отличие от DeepEval,
        # которому нужны объекты ToolCall
        tools_called=list(record.get("tools_used") or []),
        expected_tools=list(record.get("expected_tools") or []),
        perturbation_group=record.get("metamorphic_group") or None,
        name=record["id"],
    )


async def score_one(metric, case, name):
    """Считает одну метрику по одному кейсу, не роняя прогон на исключении."""
    try:
        # метрики eval_lib печатают весь разбор судьи в stdout;
        # он уже сохраняется в reason, в логе прогона он только мешает
        with contextlib.redirect_stdout(io.StringIO()):
            result = await metric.evaluate(case)
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        score = result.get("score") if isinstance(result, dict) else getattr(result, "score", None)
        reason = result.get("reason") if isinstance(result, dict) else getattr(result, "reason", None)
        return {
            "metric": name,
            "score": score,
            "success": (score is not None and score >= metric.threshold),
            "reason": reason,
            "error": None,
        }
    except Exception as exc:
        return {
            "metric": name,
            "score": None,
            "success": False,
            "reason": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run(responses_path: Path = RESPONSES, out_path: Path = OUT,
              threshold: float = THRESHOLD, limit=None):
    if not responses_path.exists():
        raise SystemExit(
            f"Нет файла {responses_path}. Сначала запустите homework_collect.py")

    records = json.loads(responses_path.read_text(encoding="utf-8"))
    if limit is not None:
        records = records[:limit]
    print(f"Кейсов к оценке: {len(records)}")

    judge = create_proxy_llm(
        model=cfg.JUDGE_MODEL,
        api_key=cfg.require("PROXY_API_KEY"),
        base_url=cfg.PROXY_BASE_URL,
    )
    print(f"Судья: {judge.get_model_name()} через {cfg.PROXY_BASE_URL}\n")

    def common_metrics():
        return [
            ("answer_relevancy", AnswerRelevancyMetric(
                model=judge, threshold=threshold, temperature=0.0)),
            ("tool_correctness", ToolCorrectnessMetric(
                threshold=threshold, exact_match=False, check_ordering=False)),
            ("task_success_rate", TaskSuccessRateMetric(
                model=judge, threshold=threshold, temperature=0.0)),
            ("tools_error", ToolsErrorMetric(
                model=judge, threshold=threshold)),
        ]

    results = []
    for i, record in enumerate(records, 1):
        case = build_case(record)
        row = {
            "id": record["id"],
            "test_type": record.get("test_type", ""),
            "metamorphic_group": record.get("metamorphic_group", ""),
            "question": record["question"],
            "tools_used": record.get("tools_used", []),
            "expected_tools": record.get("expected_tools", []),
            "agent_error": record.get("error"),
            "scores": {},
        }

        metrics = common_metrics()
        # отказ оцениваем только там, где он и ожидается
        if record.get("test_type") == "negative":
            metrics.append(("restricted_refusal", RestrictedRefusalMetric(
                threshold=0.5)))

        line = [f"[{i}/{len(records)}] {record['id']} {record.get('test_type', ''):<11}"]
        for name, metric in metrics:
            scored = await score_one(metric, case, name)
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
    print("\n" + "=" * 70)
    print("eval-ai-library: средние оценки")
    print("=" * 70)

    names = ["answer_relevancy", "tool_correctness", "task_success_rate", "tools_error"]
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

    refusal = [r for r in results if "restricted_refusal" in r["scores"]]
    if refusal:
        values = [r["scores"]["restricted_refusal"]["score"] for r in refusal
                  if r["scores"]["restricted_refusal"]["score"] is not None]
        if values:
            print(f"\nRestrictedRefusal на негативных: {sum(values) / len(values):.3f} "
                  f"по {len(values)} кейсам")

    print(f"\nДоля кейсов выше порога {threshold}:")
    for name in names:
        rows = [r for r in results if r["scores"].get(name, {}).get("score") is not None]
        if not rows:
            continue
        passed = sum(1 for r in rows if r["scores"][name]["success"])
        print(f"  {name:<18} {passed} из {len(rows)}")

    broken = [(r["id"], n, s["error"]) for r in results
              for n, s in r["scores"].items() if s["error"]]
    if broken:
        print(f"\nМетрика не посчиталась в {len(broken)} случаях:")
        for rid, name, err in broken[:10]:
            print(f"  [{rid}] {name}: {err[:70]}")


if __name__ == "__main__":
    asyncio.run(run(responses_path=RESPONSES, out_path=OUT,
                    threshold=THRESHOLD, limit=None))
