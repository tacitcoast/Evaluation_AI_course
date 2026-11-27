# homework_evaluate_multiturn.py
"""Практика 1, стадия 2: оценка записанных диалогов метриками DeepEval.

Стадия отделена от сбора: читает data/homework_dialogs.json, ничего не
спрашивает у агента. Состояние ведётся по паре (диалог, метрика), поэтому
падение на середине не теряет посчитанное.

Метрики берутся многоходовые — те, что смотрят на беседу целиком, а не на
отдельный ответ:
  TurnRelevancy            — релевантность каждого ответа своему вопросу
  RoleAdherence            — удержание роли на протяжении беседы
  KnowledgeRetention       — удержание сказанного ранее
  ConversationCompleteness — доведение задач пользователя до результата

Порог берётся из конфигурации и зафиксирован до прогона.

Запуск:
    .venv/bin/python homework_evaluate_multiturn.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "multiturn_eval"))

import homework_config as cfg
from homework_judge import ProxyJudge

from deepeval.metrics import (
    TurnRelevancyMetric,
    RoleAdherenceMetric,
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric,
)
from evaluators.deepeval.multiturn_evaluator import (
    conversation_to_test_case,
    DEFAULT_CHATBOT_ROLE,
)

METRIC_CLASSES = {
    "TurnRelevancy": TurnRelevancyMetric,
    "RoleAdherence": RoleAdherenceMetric,
    "KnowledgeRetention": KnowledgeRetentionMetric,
    "ConversationCompleteness": ConversationCompletenessMetric,
}


def load_results() -> dict:
    if cfg.MULTITURN_RESULTS_PATH.exists():
        return json.loads(cfg.MULTITURN_RESULTS_PATH.read_text(encoding="utf-8"))
    return {"threshold": cfg.METRIC_THRESHOLD, "judge": cfg.JUDGE_MODEL, "results": []}


def save(state: dict) -> None:
    cfg.DATA_DIR.mkdir(exist_ok=True)
    cfg.MULTITURN_RESULTS_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    if not cfg.DIALOGS_PATH.exists():
        raise SystemExit("Нет записанных диалогов. Сначала homework_collect_dialogs.py")

    conversations = json.loads(cfg.DIALOGS_PATH.read_text(encoding="utf-8"))["conversations"]
    state = load_results()
    done = {(r["dialog_id"], r["metric"]) for r in state["results"]}

    judge = ProxyJudge()
    print(f"Судья: {judge.get_model_name()} через {cfg.PROXY_BASE_URL}")
    print(f"Порог: {cfg.METRIC_THRESHOLD}; диалогов: {len(conversations)}; "
          f"метрик: {len(METRIC_CLASSES)}\n")

    for conv in conversations:
        test_case = conversation_to_test_case(conv, chatbot_role=DEFAULT_CHATBOT_ROLE)
        print(f"[{conv['id']}] {conv['title']}")

        for name, metric_class in METRIC_CLASSES.items():
            if (conv["id"], name) in done:
                print(f"   {name:26} уже посчитана")
                continue

            metric = metric_class(
                threshold=cfg.METRIC_THRESHOLD,
                model=judge,
                include_reason=True,
            )
            started = time.time()
            record = {
                "dialog_id": conv["id"],
                "title": conv["title"],
                "checks": conv["checks"],
                "metric": name,
                "threshold": cfg.METRIC_THRESHOLD,
            }
            try:
                metric.measure(test_case)
                record.update({
                    "score": round(float(metric.score), 4),
                    "passed": bool(metric.score >= cfg.METRIC_THRESHOLD),
                    "reason": metric.reason,
                    "error": None,
                })
                mark = "прошла" if record["passed"] else "НЕ ПРОШЛА"
                print(f"   {name:26} {record['score']:.2f}  {mark}  "
                      f"({round(time.time() - started, 1)} с)")
            except Exception as exc:
                record.update({"score": None, "passed": None, "reason": None,
                               "error": f"{type(exc).__name__}: {exc}"})
                print(f"   {name:26} ОШИБКА СУДЬИ: {record['error'][:90]}")

            state["results"].append(record)
            save(state)

    state["judge_calls"] = judge.calls
    state["judge_tokens"] = judge.prompt_tokens + judge.completion_tokens
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save(state)

    scored = [r for r in state["results"] if r.get("score") is not None]
    failed = [r for r in scored if not r["passed"]]
    print(f"\nПосчитано пар: {len(scored)}; ниже порога: {len(failed)}")
    print(f"Обращений к судье: {judge.calls}; токенов: {state['judge_tokens']}")
    print(f"Файл: {cfg.MULTITURN_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
