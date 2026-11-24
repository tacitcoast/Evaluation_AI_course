# homework_collect.py
"""Стадия 1: собрать ответы агента по датасету и сложить их в JSON.

Зачем отдельной стадией: один вопрос к агенту идёт 10-15 секунд, весь датасет
около двадцати минут. Если считать метрики в том же цикле, падение судьи на
середине обнуляет все запросы к агенту. Поэтому сбор и скоринг разделены,
результат пишется на диск после каждого вопроса, а повторный запуск
доливает только недостающие строки.

Запуск: cd Lecture5.Agents && ./.venv/bin/python homework_collect.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import homework_config as cfg                       # noqa: E402
from agent_connector import AgentConnector          # noqa: E402
from dataset_parser import DatasetParser            # noqa: E402

DATASET = "data/homework_dataset_full.xlsx"
OUT = Path("data/agent_responses.json")


def load_done(path: Path) -> dict:
    """Читает уже собранные ответы, чтобы не спрашивать агента дважды."""
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        return {r["id"]: r for r in records}
    except Exception as exc:
        print(f"Не удалось прочитать {path}: {exc}. Начинаю заново.")
        return {}


def collect(dataset_path: str = DATASET, out_path: Path = OUT,
            sleep_time: float = 1.0, limit=None):
    api_key = cfg.require("AGENT_API_KEY")

    parser = DatasetParser()
    df = parser.load_dataset(dataset_path)
    if df is None:
        print("Датасет не загрузился")
        return

    pairs = parser.get_question_response_pairs(df)
    test_types = df["test_type"].tolist() if "test_type" in df.columns else [""] * len(pairs)
    if "metamorphic_group" in df.columns:
        groups = df["metamorphic_group"].fillna("").tolist()
    else:
        groups = [""] * len(pairs)

    done = load_done(out_path)
    print(f"Уже собрано: {len(done)} из {len(pairs)}")

    agent = AgentConnector(
        endpoint_url=cfg.AGENT_URL,
        api_key=api_key,
        user_id=cfg.AGENT_USER_ID,
        timeout=120,
    )
    print(f"Сессия агента: {agent.session_id}\n")

    records = list(done.values())
    asked = 0

    for i, pair in enumerate(pairs):
        rid = f"{i:03d}"
        if rid in done:
            continue
        if limit is not None and asked >= limit:
            print(f"\nДостигнут лимит в {limit} запросов, останавливаюсь.")
            break

        question = pair["question"]
        urls = pair["urls"]

        print(f"[{i + 1}/{len(pairs)}] {test_types[i]:<11} {question[:52]}")

        started = time.time()
        response = agent.query(question, urls=urls or None)
        elapsed = round(time.time() - started, 2)

        error = response.get("error")
        record = {
            "id": rid,
            "index": i,
            "test_type": test_types[i],
            "metamorphic_group": groups[i],
            "question": question,
            "urls": urls,
            "expected_response": pair["expected_response"],
            "expected_tools": pair["expected_tools"],
            "output": response.get("output", ""),
            "tools_used": response.get("tools_used", []),
            "error": error,
            "latency_sec": elapsed,
        }
        records.append(record)
        asked += 1

        if error:
            print(f"    ОШИБКА за {elapsed} с: {error[:90]}")
        else:
            tools = ", ".join(record["tools_used"]) or "ни одного"
            print(f"    {elapsed} с, инструменты: {tools}")
            print(f"    ответ: {str(record['output'])[:110]}")

        # пишем после каждого вопроса, чтобы не потерять прогресс
        records.sort(key=lambda r: r["index"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        time.sleep(sleep_time)

    print(f"\nВ файле {out_path}: {len(records)} записей, из них новых {asked}")
    errors = [r for r in records if r.get("error")]
    print(f"С ошибками: {len(errors)}")
    for r in errors:
        print(f"   [{r['id']}] {str(r['error'])[:70]}  <- {r['question'][:40]}")


if __name__ == "__main__":
    # limit=None собирает весь датасет; поставьте число, чтобы прогнать часть
    collect(dataset_path=DATASET, out_path=OUT, sleep_time=1.0, limit=None)
