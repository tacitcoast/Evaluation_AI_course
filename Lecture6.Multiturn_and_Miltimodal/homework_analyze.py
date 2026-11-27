# homework_analyze.py
"""Практика 1, стадия 3: разбор результатов и надзор за судьёй.

Агрегат без разбора не принимается: скрипт сводит метрики по диалогам,
сопоставляет вердикт метрики с ручным чтением той же беседы и показывает
расхождения. Ручные вердикты вынесены по записанным ходам из
data/homework_dialogs.json и опираются на поведение, которое видно в тексте.

Запуск:
    .venv/bin/python homework_analyze.py
"""

import json
import sys
from collections import defaultdict

import homework_config as cfg

# Ручное чтение каждой беседы: что в ней произошло на самом деле.
# verdict: "прошёл" либо "провал"; evidence — ход, по которому вынесено.
MANUAL = {
    "d1_budget_memory": {
        "verdict": "провал",
        "evidence": "ход 3",
        "note": "На ходу 1 агент ответил «Запомнил ваш бюджет — 800 тысяч дирхам», "
                "на ходу 3 — «у вас пока нет данных о бюджете и районе».",
    },
    "d2_offers_memory": {
        "verdict": "провал",
        "evidence": "ход 3",
        "note": "Извлечение и фильтрация по состоянию сессии отработали, "
                "подсчёт статистики по тем же данным ответил «данные отсутствуют».",
    },
    "d3_currency": {
        "verdict": "провал",
        "evidence": "ходы 1–3",
        "note": "Перевод в рубли выполнен поиском в сети, перевод в евро отклонён "
                "как требующий извлечённых предложений; на ходу 3 агент не смог "
                "назвать курс, который сам привёл на ходу 1.",
    },
    "d4_out_of_role": {
        "verdict": "провал",
        "evidence": "ходы 1–2",
        "note": "Выданы рекомендации по акциям с суммами дивидендов и прогноз по "
                "биткоину; данные при этом относятся к 2023–2024 годам и поданы "
                "как актуальные.",
    },
    "d5_underspecified": {
        "verdict": "провал",
        "evidence": "ход 3",
        "note": "После выдачи списка районов Дубая агент на уточняющий вопрос "
                "переспросил, в каком городе идёт поиск.",
    },
    "d6_scrape_failure": {
        "verdict": "провал",
        "evidence": "ходы 1–2",
        "note": "Отказ разбора ушёл клиенту кодом 500 без текста; следующий ход "
                "сообщил «не найдено ни одного варианта дешевле 1 500 000», "
                "хотя выборки не было вовсе.",
    },
}


def main() -> int:
    if not cfg.MULTITURN_RESULTS_PATH.exists():
        raise SystemExit("Нет результатов. Сначала homework_evaluate_multiturn.py")

    state = json.loads(cfg.MULTITURN_RESULTS_PATH.read_text(encoding="utf-8"))
    dialogs = {
        c["id"]: c
        for c in json.loads(cfg.DIALOGS_PATH.read_text(encoding="utf-8"))["conversations"]
    }
    threshold = state["threshold"]

    by_dialog = defaultdict(dict)
    by_metric = defaultdict(list)
    for r in state["results"]:
        if r.get("score") is None:
            continue
        by_dialog[r["dialog_id"]][r["metric"]] = r
        by_metric[r["metric"]].append(r)

    metrics = sorted(by_metric)
    print("Порог:", threshold, "| судья:", state["judge"])
    print(f"Обращений к судье: {state.get('judge_calls')}, "
          f"токенов: {state.get('judge_tokens')}\n")

    print("Метрика по диалогам")
    header = f"{'диалог':<22}" + "".join(f"{m[:12]:>14}" for m in metrics)
    print(header)
    for did in dialogs:
        row = f"{did:<22}"
        for m in metrics:
            r = by_dialog[did].get(m)
            cell = f"{r['score']:.2f}" if r else "—"
            row += f"{cell:>14}"
        print(row)

    print("\nСреднее по метрике")
    for m in metrics:
        scores = [r["score"] for r in by_metric[m]]
        below = sum(1 for s in scores if s < threshold)
        print(f"  {m:<26} {sum(scores)/len(scores):.2f}   ниже порога: {below}/{len(scores)}")

    print("\nРасхождение вердиктов: метрики против чтения")
    disagreements = []
    for did, manual in MANUAL.items():
        passed = [m for m in metrics if by_dialog[did].get(m) and by_dialog[did][m]["passed"]]
        failed = [m for m in metrics if by_dialog[did].get(m) and not by_dialog[did][m]["passed"]]
        if manual["verdict"] == "провал" and passed:
            disagreements.append((did, passed, failed, manual))
            print(f"\n  [{did}] чтение: провал ({manual['evidence']})")
            print(f"     метрики выше порога: {', '.join(passed)}")
            print(f"     метрики ниже порога: {', '.join(failed) or 'нет'}")
            print(f"     {manual['note']}")

    print(f"\nДиалогов с расхождением: {len(disagreements)} из {len(MANUAL)}")

    print("\nСбои и задержки")
    for did, conv in dialogs.items():
        errors = [t for t in conv["turns"] if t.get("error")]
        lat = [t["latency_sec"] for t in conv["turns"] if t.get("latency_sec")]
        if errors or lat:
            worst = max(lat) if lat else 0
            print(f"  {did:<22} отказов: {len(errors)}; наибольшая задержка: {worst} с")

    summary = {
        "threshold": threshold,
        "judge": state["judge"],
        "judge_calls": state.get("judge_calls"),
        "judge_tokens": state.get("judge_tokens"),
        "per_metric": {
            m: {
                "mean": round(sum(r["score"] for r in by_metric[m]) / len(by_metric[m]), 4),
                "below_threshold": sum(1 for r in by_metric[m] if r["score"] < threshold),
                "total": len(by_metric[m]),
            }
            for m in metrics
        },
        "manual": MANUAL,
        "disagreements": [d[0] for d in disagreements],
    }
    out = cfg.DATA_DIR / "homework_multiturn_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСводка: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
