# homework_collect_dialogs.py
"""Практика 1, стадия 1: запись диалогов с агентом.

Сбор ответов и их оценка разведены на две стадии со своим состоянием.
Падение на середине сбора не теряет уже записанное: файл дописывается
после каждого диалога.

Запуск из каталога лекции:
    .venv/bin/python homework_collect_dialogs.py
"""

import json
import sys
import time
from datetime import datetime

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "multiturn_eval"))

import homework_config as cfg
from homework_dialogs import DIALOGS
from agent_connector import AgentConnector

PAUSE_BETWEEN_TURNS = 1.0
MAX_ATTEMPTS = 3


def load_done() -> dict:
    if cfg.DIALOGS_PATH.exists():
        return json.loads(cfg.DIALOGS_PATH.read_text(encoding="utf-8"))
    return {"conversations": []}


def save(state: dict) -> None:
    cfg.DATA_DIR.mkdir(exist_ok=True)
    cfg.DIALOGS_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def ask(agent: AgentConnector, question: str) -> dict:
    """Один ход. Сбой стенда повторяется, отказ агента — результат."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.time()
        result = agent.query(question)
        elapsed = round(time.time() - started, 2)
        error = result.get("error")
        if error and attempt < MAX_ATTEMPTS:
            print(f"      сбой стенда ({error}), попытка {attempt + 1}")
            time.sleep(2 * attempt)
            continue
        return {
            "content": result.get("output", ""),
            "tools_called": result.get("tools_used", []),
            "latency_sec": elapsed,
            "attempts": attempt,
            "error": error,
        }
    return {"content": "", "tools_called": [], "latency_sec": 0.0,
            "attempts": MAX_ATTEMPTS, "error": "исчерпаны попытки"}


def main() -> int:
    cfg.require("AGENT_API_KEY")
    state = load_done()
    recorded = {c["id"] for c in state["conversations"]}

    for spec in DIALOGS:
        if spec["id"] in recorded:
            print(f"[{spec['id']}] уже записан, пропуск")
            continue

        agent = AgentConnector(
            endpoint_url=cfg.AGENT_URL,
            api_key=cfg.AGENT_API_KEY,
            user_id=cfg.AGENT_USER_ID,
        )
        print(f"\n[{spec['id']}] {spec['title']}")
        print(f"   сессия {agent.session_id}")

        turns = []
        for number, question in enumerate(spec["turns"], 1):
            print(f"   ход {number}: {question[:70]}")
            turns.append({"role": "user", "content": question})
            answer = ask(agent, question)
            mark = "ОШИБКА" if answer["error"] else f"{answer['latency_sec']} с"
            tools = ", ".join(answer["tools_called"]) or "без инструментов"
            print(f"      {mark}; {tools}; {len(answer['content'])} знаков")
            turns.append({
                "role": "assistant",
                "content": answer["content"],
                "tools_called": answer["tools_called"],
                "latency_sec": answer["latency_sec"],
                "attempts": answer["attempts"],
                "error": answer["error"],
            })
            time.sleep(PAUSE_BETWEEN_TURNS)

        state["conversations"].append({
            "id": spec["id"],
            "title": spec["title"],
            "checks": spec["checks"],
            "expected": spec["expected"],
            "failure": spec["failure"],
            "session_id": agent.session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "turns_count": len(turns),
            "turns": turns,
        })
        save(state)

    total_turns = sum(c["turns_count"] for c in state["conversations"])
    print(f"\nЗаписано диалогов: {len(state['conversations'])}, ходов: {total_turns}")
    print(f"Файл: {cfg.DIALOGS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
