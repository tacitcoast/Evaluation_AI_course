# homework_trace_send_evaliqa.py
"""Практика 2: отправка трейсов агента в EvaliQA для онлайн-оценки.

Онлайн-оценка в EvaliQA считает метрики по трейсам, которые приходят из
работающей системы, а не по заранее собранному датасету. Поэтому трейсы надо
сначала прислать, и только потом к ним привязываются метрики.

Шлём два вида трейсов одного и того же агента, чтобы было видно разницу.

  hosted — боевой агент по HTTP. /ask возвращает только имена инструментов,
           поэтому у дочерних шагов нет ни аргументов, ни результатов.
           Это трейс системы под тестом, как он доступен снаружи.

  local  — тот же код агента локально, трассируется штатным колбэком
           eval_lib для LangChain. В трейс попадают вызовы модели с токенами
           и вызовы инструментов с аргументами, результатами и ошибками.

Настройки берутся из .env: TRACING_ENABLED, TRACING_URL, TRACING_PROJECT,
TRACING_API_KEY — их выдаёт EvaliQA на странице Tracing.

Запуск:
    ./.venv-agent/bin/python homework_trace_send_evaliqa.py hosted
    ./.venv-agent/bin/python homework_trace_send_evaliqa.py local
    ./.venv-agent/bin/python homework_trace_send_evaliqa.py both
"""

import json
import os
import sys
import time
import types
import uuid
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import homework_config as cfg                                    # noqa: E402

# Конфигурация трассировки читается при импорте eval_lib.tracing,
# поэтому проверяем её до импорта.
for _name in ("TRACING_ENABLED", "TRACING_URL", "TRACING_PROJECT", "TRACING_API_KEY"):
    cfg.require(_name)

from eval_lib.tracing import tracer, SpanType                    # noqa: E402

RESPONSES = Path("data/agent_responses.json")
OUT = Path("data/traces_sent_evaliqa.json")

USER_ID = cfg.AGENT_USER_ID


def pick_hosted_cases(records, n_ok=7, n_failed=6, n_special=2):
    """Отбирает подмножество датасета так, чтобы в трейсах были оба исхода.

    Метрики онлайн-оценки интересно смотреть, когда среди трейсов есть и
    удачные прогоны, и отказы. Берём и те, и другие, плюс два особых случая:
    вопрос не по теме и попытку инъекции промпта.
    """
    ok = [r for r in records if not r.get("error")]
    failed = [r for r in records if r.get("error")]

    def special(record):
        text = record["question"].lower()
        return ("погод" in text or "инструкц" in text or "игнорируй" in text
                or "забудь" in text)

    chosen, seen = [], set()
    for record in failed[:n_failed] + ok[:n_ok]:
        if record["id"] not in seen:
            chosen.append(record)
            seen.add(record["id"])
    for record in records:
        if n_special <= 0:
            break
        if special(record) and record["id"] not in seen:
            chosen.append(record)
            seen.add(record["id"])
            n_special -= 1
    return chosen


def trace_hosted(limit=None):
    """Боевой агент: один трейс на запрос, дочерний шаг на каждый инструмент."""
    from agent_connector import AgentConnector

    if not RESPONSES.exists():
        raise SystemExit(f"Нет {RESPONSES}: сначала homework_collect.py")
    records = json.loads(RESPONSES.read_text(encoding="utf-8"))
    cases = pick_hosted_cases(records)
    if limit:
        cases = cases[:limit]

    agent = AgentConnector(
        endpoint_url=cfg.AGENT_URL,
        api_key=cfg.require("AGENT_API_KEY"),
        user_id=USER_ID,
        timeout=120,
    )
    session_id = agent.session_id

    print(f"Боевой агент: {len(cases)} запросов, сессия {session_id}\n")
    sent = []
    for i, record in enumerate(cases, 1):
        question = record["question"]
        urls = record.get("urls") or []

        trace_id = tracer.start_trace("housing_price_agent")
        started = time.time()

        root = tracer.start_span(
            "ask",
            SpanType.AGENT_STEP,
            input_data={"question": question, "urls": urls},
            metadata={"source": "hosted", "case_id": record["id"],
                      "test_type": record.get("test_type", "")},
        )

        result = agent.query(question, urls or None)
        elapsed = round(time.time() - started, 2)
        error = result.get("error")
        output = result.get("output") or ""
        tools = result.get("tools_used") or []

        # Шаг на каждый инструмент. Аргументов и результатов сервис не отдаёт:
        # именно этот пробел и показывает разницу с локальным трейсом.
        for position, name in enumerate(tools, 1):
            step = tracer.start_span(
                name,
                SpanType.TOOL_CALL,
                input_data=None,
                metadata={"position": position,
                          "arguments_available": False,
                          "result_available": False},
                parent_span_id=root.span_id if root else None,
                set_current=False,
            )
            tracer.end_span(step, output=None, status="success")

        if error:
            tracer.end_span(root, error=str(error), error_type="HTTPError")
        else:
            tracer.end_span(root, output=output)

        tracer.set_trace_metadata(
            model="hosted-agent",
            input=question if not urls else f"{question}\nСсылки: {', '.join(urls)}",
            output=output or f"[сбой сервиса] {error}",
            response_time=elapsed,
            session_id=session_id,
            user_id=USER_ID,
        )
        tracer.end_trace()

        mark = "ОШИБКА" if error else "ok    "
        print(f"[{i}/{len(cases)}] {mark} {record['id']:<6} {elapsed:>6.2f} с  "
              f"инструментов {len(tools)}  {question[:44]}")
        sent.append({"trace_id": trace_id, "source": "hosted", "case_id": record["id"],
                     "question": question, "urls": urls, "tools_used": tools,
                     "error": error, "latency_sec": elapsed,
                     "test_type": record.get("test_type", "")})
        time.sleep(1.0)
    return sent


def _install_langfuse_stub():
    """agent.py импортирует langfuse: в чужой проект писать не нужно."""
    class _Noop:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def observe(*a, **k):
        def wrap(fn):
            return fn
        return wrap

    langfuse = types.ModuleType("langfuse")
    langfuse.observe = observe
    langfuse.get_client = lambda: types.SimpleNamespace(
        start_as_current_observation=lambda **k: _Noop())
    langfuse.propagate_attributes = lambda **k: _Noop()
    langfuse_langchain = types.ModuleType("langfuse.langchain")
    langfuse_langchain.CallbackHandler = _Noop
    sys.modules["langfuse"] = langfuse
    sys.modules["langfuse.langchain"] = langfuse_langchain


# Те же вопросы, что в homework_trace_local.py: сравнимо с отчётом Практики 1.
BOOKS = "http://books.toscrape.com/"
SHOP = "https://webscraper.io/test-sites/e-commerce/allinone"
URL404 = "https://www.propertyfinder.ae/en/no-such-page-xyz-404"

LOCAL_CASES = [
    ("L01", "Извлеки товары с этой страницы", [BOOKS]),
    ("L02", "Извлеки товары и посчитай статистику по ценам", [BOOKS]),
    ("L03", "Отбери товары дешевле 20 и посчитай по ним среднее", [BOOKS]),
    ("L04", "Переведи цены в рубли", [BOOKS]),
    ("L05", "Переведи цены в евро", [SHOP]),
    ("L06", "Переведи цены в рубли", [SHOP]),
    ("L07", "Извлеки объявления с этой страницы", [URL404]),
    ("L08", "Отбери товары дешевле 100 и посчитай среднее", [BOOKS]),
]


def trace_local(limit=None):
    """Локальный агент под штатным колбэком eval_lib для LangChain."""
    os.environ["OPENAI_API_KEY"] = cfg.require("PROXY_API_KEY")
    os.environ["OPENAI_BASE_URL"] = cfg.PROXY_BASE_URL
    os.environ["OPENAI_API_BASE"] = cfg.PROXY_BASE_URL
    os.environ["OPENAI_MODEL"] = cfg.JUDGE_MODEL
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ.pop("LANGCHAIN_API_KEY", None)
    os.environ.pop("LANGSMITH_API_KEY", None)
    _install_langfuse_stub()

    sys.path.insert(0, str(HERE / "ai-agent_l5"))
    from app.agent import agent as A
    from eval_lib.tracing import EvalLibCallbackHandler

    handler = EvalLibCallbackHandler()
    session_id = f"local-{uuid.uuid4().hex[:8]}"
    cases = LOCAL_CASES[:limit] if limit else LOCAL_CASES

    print(f"\nЛокальный агент: {len(cases)} прогонов, сессия {session_id}\n")
    sent = []
    for i, (case_id, question, urls) in enumerate(cases, 1):
        A._SESSION_STATE["offers"] = []
        trace_id = tracer.start_trace("housing_price_agent_local")
        started = time.time()
        error, output = None, ""
        try:
            result = A.executor.invoke(
                {"question": question, "urls": urls, "max_items": 50},
                config={"callbacks": [handler]},
            )
            output = result.get("output", "")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round(time.time() - started, 2)

        tracer.set_trace_metadata(
            model=cfg.JUDGE_MODEL,
            input=f"{question}\nСсылки: {', '.join(urls) or 'нет'}",
            output=output or f"[исключение] {error}",
            response_time=elapsed,
            session_id=session_id,
            user_id=USER_ID,
        )
        tracer.end_trace()

        mark = "ОШИБКА" if error else "ok    "
        print(f"[{i}/{len(cases)}] {mark} {case_id} {elapsed:>6.2f} с  {question[:44]}")
        if error:
            print(f"        {error[:110]}")
        sent.append({"trace_id": trace_id, "source": "local", "case_id": case_id,
                     "question": question, "urls": urls, "error": error,
                     "latency_sec": elapsed, "test_type": "local_trace"})
    return sent


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print("=" * 72)
    print("Отправка трейсов в EvaliQA")
    print(f"Проект: {os.environ['TRACING_PROJECT']}")
    print(f"Приёмник: {os.environ['TRACING_URL']}")
    print("=" * 72)

    sent = []
    if mode in ("hosted", "both"):
        sent += trace_hosted(limit)
    if mode in ("local", "both"):
        sent += trace_local(limit)

    tracer.flush()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    OUT.write_text(json.dumps(existing + sent, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nОтправлено трейсов: {len(sent)}. Список: {OUT}")
    print("Трейсы появляются в EvaliQA на вкладке Tracing примерно за минуту.")


if __name__ == "__main__":
    main()
