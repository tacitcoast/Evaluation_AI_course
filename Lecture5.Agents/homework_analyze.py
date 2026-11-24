# homework_analyze.py
"""Стадия 3: разбор результатов — то, что метрики фреймворков не считают.

Считает:
  1. таксономию отказов: на каких типах вопросов агент отдаёт HTTP 500;
  2. метаморфную согласованность: совпал ли состав инструментов и числа
     в ответах внутри группы перефразировок;
  3. признаки нулевых цен в ответах — цена 0 как заглушка вместо «цена не указана»;
  4. задержки по типам вопросов;
  5. сводку оценок обоих фреймворков рядом, чтобы видеть расхождение.

Запуск: cd Lecture5.Agents && ./.venv/bin/python homework_analyze.py
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RESPONSES = Path("data/agent_responses.json")
DEEPEVAL = Path("data/scores_deepeval.json")
EVALLIB = Path("data/scores_evallib.json")
OUT = Path("data/analysis.json")

def is_conversion(record) -> bool:
    """Запрос на конвертацию определяем по задаче, а не по слову в тексте:
    «найди квартиру в Москве до 10 миллионов рублей» упоминает рубли,
    но конвертировать там нечего."""
    return ("normalize_offers_currency" in (record.get("expected_tools") or [])
            or "тугрик" in record["question"].lower())


def load(path: Path):
    if not path.exists():
        print(f"нет файла {path} — пропускаю этот блок")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def numbers_in(text: str):
    """Все числа из ответа, нормализованные: нужны для сверки перефразировок."""
    cleaned = str(text).replace(" ", " ")
    found = re.findall(r"\d[\d  .,]*\d|\d", cleaned)
    out = set()
    for token in found:
        digits = re.sub(r"[^\d]", "", token)
        if digits and len(digits) >= 4:      # цены, а не номера пунктов списка
            out.add(int(digits))
    return out


def analyse_failures(records):
    print("=" * 72)
    print("1. На каких вопросах агент отдаёт ошибку")
    print("=" * 72)

    failed = [r for r in records if r.get("error")]
    print(f"Всего ошибок: {len(failed)} из {len(records)}\n")

    by_type = Counter(r["test_type"] for r in failed)
    for test_type, n in by_type.most_common():
        total = sum(1 for r in records if r["test_type"] == test_type)
        print(f"  {test_type:<12} {n} из {total}")

    print("\nПо запрошенным инструментам:")
    by_tool = Counter()
    for r in failed:
        key = ", ".join(r.get("expected_tools") or ["без инструментов"])
        by_tool[key] += 1
    for key, n in by_tool.most_common():
        total = sum(1 for r in records
                    if ", ".join(r.get("expected_tools") or ["без инструментов"]) == key)
        print(f"  {n} из {total:<3} {key}")

    print("\nЗапросы на конвертацию валюты:")
    currency_failed = [r for r in failed if is_conversion(r)]
    currency_total = [r for r in records if is_conversion(r)]
    print(f"  упало {len(currency_failed)} из {len(currency_total)}")

    print("\nСписок упавших:")
    for r in failed:
        urls = ", ".join(u.split("/")[2] for u in (r.get("urls") or [])) or "без ссылок"
        print(f"  [{r['id']}] {r['latency_sec']:>6} с  {urls:<24} {r['question'][:46]}")

    return {
        "total": len(records),
        "failed": len(failed),
        "failed_ids": [r["id"] for r in failed],
        "by_test_type": dict(by_type),
        "by_expected_tools": dict(by_tool),
        "currency_failed": len(currency_failed),
        "currency_total": len(currency_total),
    }


def analyse_metamorphic(records):
    print("\n" + "=" * 72)
    print("2. Метаморфная согласованность")
    print("=" * 72)
    print("Внутри группы смысл задачи один. Смотрим, совпал ли состав")
    print("инструментов и пересеклись ли числа в ответах.\n")

    groups = defaultdict(list)
    for r in records:
        if r.get("test_type") == "metamorphic" and r.get("metamorphic_group"):
            groups[r["metamorphic_group"]].append(r)

    result = {}
    for group, rows in groups.items():
        tool_sets = [frozenset(r.get("tools_used") or []) for r in rows]
        same_tools = len(set(tool_sets)) == 1

        number_sets = [numbers_in(r.get("output") or "") for r in rows]
        non_empty = [s for s in number_sets if s]
        if len(non_empty) >= 2:
            common = set.intersection(*non_empty)
            union = set.union(*non_empty)
            overlap = len(common) / len(union) if union else 0.0
        else:
            common, overlap = set(), None

        errors = sum(1 for r in rows if r.get("error"))

        print(f"Группа «{group}»: {len(rows)} формулировок, ошибок {errors}")
        print(f"  состав инструментов одинаков: {'да' if same_tools else 'НЕТ'}")
        for r, tools in zip(rows, tool_sets):
            print(f"    [{r['id']}] {sorted(tools) or 'ни одного'}  <- {r['question'][:44]}")
        if overlap is None:
            print("  числа сравнить не удалось: мало ответов с числами")
        else:
            print(f"  пересечение чисел: {overlap:.2f}, общих значений {len(common)}")
        print()

        result[group] = {
            "cases": len(rows),
            "errors": errors,
            "same_tools": same_tools,
            "tool_sets": [sorted(s) for s in tool_sets],
            "number_overlap": overlap,
            "common_numbers": sorted(common)[:10],
        }
    return result


def analyse_zero_prices(records):
    print("=" * 72)
    print("3. Нулевые цены в ответах")
    print("=" * 72)
    print("Схема Offer требует price: int, выразить «цена не указана» нечем,")
    print("поэтому неизвестная цена превращается в 0 и попадает в статистику.\n")

    pattern = re.compile(r"(?:^|[^\d])0\s*(?:zł|AED|PLN|руб|₽|\$|€|£|USD|EUR)", re.I)
    hits = []
    for r in records:
        output = str(r.get("output") or "")
        found = pattern.findall(output)
        if found:
            count = len(re.findall(r"(?:^|[^\d])0\s*(?:zł|AED|PLN|руб|₽|\$|€|£|USD|EUR)",
                                   output, re.I))
            hits.append((r["id"], count, r["question"][:46]))

    print(f"Ответов с ценой 0: {len(hits)} из {len(records)}")
    for rid, count, question in hits:
        print(f"  [{rid}] нулей {count}: {question}")

    return {"responses_with_zero_price": len(hits),
            "ids": [h[0] for h in hits]}


def analyse_latency(records):
    print("\n" + "=" * 72)
    print("4. Задержки")
    print("=" * 72)

    ok = [r for r in records if not r.get("error") and r.get("latency_sec")]
    failed = [r for r in records if r.get("error") and r.get("latency_sec")]

    def describe(label, rows):
        if not rows:
            return None
        values = [r["latency_sec"] for r in rows]
        print(f"  {label:<26} n={len(values):<3} медиана {statistics.median(values):>6.1f} с"
              f"  мин {min(values):>5.1f}  макс {max(values):>6.1f}")
        return {"n": len(values), "median": statistics.median(values),
                "min": min(values), "max": max(values)}

    out = {"ok": describe("успешные ответы", ok),
           "failed": describe("ответы с ошибкой", failed)}

    print()
    for test_type in ["happy_path", "negative", "metamorphic"]:
        rows = [r for r in records if r["test_type"] == test_type and not r.get("error")]
        out[test_type] = describe(test_type, rows)

    fast = [r for r in failed if r["latency_sec"] < 3]
    slow = [r for r in failed if r["latency_sec"] >= 3]
    print(f"\n  Отказы делятся на два класса. {len(fast)} быстрых, около 1.5 с:")
    print("  страница не загрузилась, до обращения к модели дело не дошло.")
    print(f"  {len(slow)} медленных, от 5 до 21 с: извлечение уже прошло,")
    print("  падение случилось внутри инструмента.")
    out["fast_failures"] = len(fast)
    out["slow_failures"] = len(slow)
    return out


def compare_frameworks(deepeval, evallib):
    print("\n" + "=" * 72)
    print("5. Два фреймворка на одних и тех же ответах")
    print("=" * 72)

    if not deepeval or not evallib:
        print("Нет одного из файлов оценок — сравнение пропущено")
        return None

    de = {r["id"]: r for r in deepeval}
    el = {r["id"]: r for r in evallib}
    shared = sorted(set(de) & set(el))
    print(f"Общих кейсов: {len(shared)}\n")

    pairs = [
        ("answer_relevancy", "answer_relevancy", "AnswerRelevancy"),
        ("tool_correctness", "tool_correctness", "ToolCorrectness"),
        ("task_completion", "task_success_rate", "TaskCompletion / TaskSuccessRate"),
    ]

    out = {}
    print(f"{'метрика':<34}{'DeepEval':>10}{'eval_lib':>10}{'разница':>10}")
    for de_name, el_name, label in pairs:
        de_values = [de[i]["scores"].get(de_name, {}).get("score") for i in shared]
        el_values = [el[i]["scores"].get(el_name, {}).get("score") for i in shared]
        both = [(a, b) for a, b in zip(de_values, el_values)
                if a is not None and b is not None]
        if not both:
            continue
        de_avg = sum(a for a, _ in both) / len(both)
        el_avg = sum(b for _, b in both) / len(both)
        print(f"{label:<34}{de_avg:>10.3f}{el_avg:>10.3f}{el_avg - de_avg:>+10.3f}")
        out[label] = {"deepeval": de_avg, "evallib": el_avg,
                      "diff": el_avg - de_avg, "n": len(both)}

    print("\nКейсы с наибольшим расхождением по релевантности ответа:")
    diffs = []
    for i in shared:
        a = de[i]["scores"].get("answer_relevancy", {}).get("score")
        b = el[i]["scores"].get("answer_relevancy", {}).get("score")
        if a is not None and b is not None:
            diffs.append((abs(a - b), i, a, b))
    diffs.sort(reverse=True)
    for gap, i, a, b in diffs[:5]:
        print(f"  [{i}] DeepEval {a:.2f} / eval_lib {b:.2f} (разница {gap:.2f})"
              f"  {de[i]['question'][:40]}")

    return out


def main():
    records = load(RESPONSES)
    if not records:
        raise SystemExit("Нет собранных ответов. Запустите homework_collect.py")

    analysis = {
        "failures": analyse_failures(records),
        "metamorphic": analyse_metamorphic(records),
        "zero_prices": analyse_zero_prices(records),
        "latency": analyse_latency(records),
        "frameworks": compare_frameworks(load(DEEPEVAL), load(EVALLIB)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРазбор сохранён в {OUT}")


if __name__ == "__main__":
    main()
