# homework_trace_local.py
"""Локальный прогон агента из ai-agent_l5 со снятием полного трейса.

Зачем. Боевой /ask отдаёт только имена инструментов, поэтому метрики по трейсам
в EvaliQA на нём ничего не видят. Здесь тот же код агента поднимается локально,
каждый вызов инструмента оборачивается замером, и в трейс попадают аргументы
вызова, результат, длительность и исключение, если оно было.

Важная оговорка. Это код из репозитория, а развёрнутый агент работает не на нём:
локальное extract_offers на propertyfinder возвращает ноль объявлений, потому что
первые 160 000 символов там почти целиком <script>, который strip_boilerplate
выбрасывает, а боевой агент объявления отдаёт. Таблица курсов в tools_ext.py при
этом совпадает с боевой: она пересчитывает доллар в рубль ровно по 90. Поэтому
локальный трейс годится как свидетельство про инструменты и про их аргументы,
но не про скрейпинг.

Модель-судья здесь не участвует: сам агент ходит в ту же прокси курса.
Langfuse и LangSmith заглушены, чтобы не писать в чужие проекты.

Запуск: cd Lecture5.Agents && ./.venv-agent/bin/python homework_trace_local.py
"""

import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import homework_config as cfg                                   # noqa: E402

# --- агент ходит в прокси курса, а не напрямую в OpenAI -----------------
os.environ["OPENAI_API_KEY"] = cfg.require("PROXY_API_KEY")
os.environ["OPENAI_BASE_URL"] = cfg.PROXY_BASE_URL
# ChatOpenAI читает OPENAI_API_BASE, клиент openai — OPENAI_BASE_URL
os.environ["OPENAI_API_BASE"] = cfg.PROXY_BASE_URL
os.environ["OPENAI_MODEL"] = cfg.JUDGE_MODEL

# LangSmith подменять нельзя: от него зависит сам langchain_core.
# Выключаем отправку, тогда @traceable работает как обычный декоратор.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)
os.environ.pop("LANGSMITH_API_KEY", None)


# --- заглушки трассировки ------------------------------------------------
# agent.py импортирует langfuse, писать в чужой проект нам не нужно.
# langsmith при этом настоящий: его подмена рушит langchain_core.
def _install_stubs():
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


_install_stubs()

AGENT_DIR = HERE / "ai-agent_l5"
sys.path.insert(0, str(AGENT_DIR))

from app.agent import agent as A           # noqa: E402

OUT = Path("data/traces_local.json")
MAX_VALUE_CHARS = 600


def shorten(value):
    """Укорачивает значения для трейса, сохраняя все ключи.

    Объёмный список объявлений сворачивается в счётчик с первым элементом,
    а скалярные аргументы вроде max_price и target_currency остаются целиком:
    именно по ним видно, донёс ли агент ограничение пользователя до инструмента.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            return {"items": 0}
        if len(value) <= 2:
            return [shorten(v) for v in value]
        return {"items": len(value), "first": shorten(value[0])}
    if isinstance(value, dict):
        return {k: shorten(v) for k, v in value.items()}
    text = str(value)
    return text if len(text) <= MAX_VALUE_CHARS else text[:MAX_VALUE_CHARS] + "..."


def instrument(tools, sink: list):
    """Оборачивает функции инструментов замером, не меняя их поведения."""
    for tool in tools:
        original = tool.func
        name = tool.name

        def make(original=original, name=name):
            def wrapper(*args, **kwargs):
                started = time.time()
                step = {
                    "step_id": str(len(sink) + 1),
                    # словарь типов задан в eval_lib.tracing.SpanType:
                    # метрики по трейсам считают шаги с типом tool_call,
                    # на "tool" они шаг просто не замечают
                    "type": "tool_call",
                    "name": name,
                    "input": shorten(kwargs or {"args": args}),
                    "output": None,
                    "duration_ms": None,
                    "status": "success",
                    "error": None,
                    "error_type": None,
                }
                try:
                    result = original(*args, **kwargs)
                except Exception as exc:
                    step["duration_ms"] = round((time.time() - started) * 1000, 1)
                    step["status"] = "error"
                    step["error"] = str(exc)[:300]
                    step["error_type"] = type(exc).__name__
                    sink.append(step)
                    raise
                step["duration_ms"] = round((time.time() - started) * 1000, 1)
                step["output"] = shorten(result)
                if isinstance(result, list):
                    step["metadata"] = {"items": len(result)}
                sink.append(step)
                return result
            return wrapper

        tool.func = make()


# Вопросы подобраны так, чтобы в трейсе было видно каждый механизм.
# Основная площадка — books.toscrape.com: страница 50 КБ, цены в HTML,
# прогон дешёвый и повторяемый. Цены там в фунтах, которых нет в таблице курсов.
BOOKS = "http://books.toscrape.com/"
SHOP = "https://webscraper.io/test-sites/e-commerce/allinone"
PF = "https://www.propertyfinder.ae/en/buy/dubai/apartments-for-sale.html"
URL404 = "https://www.propertyfinder.ae/en/no-such-page-xyz-404"

CASES = [
    ("L01", "Извлеки товары с этой страницы", [BOOKS], ["extract_offers"],
     "базовый случай: один инструмент"),
    ("L02", "Извлеки товары и посчитай статистику по ценам", [BOOKS],
     ["extract_offers", "compute_stats"], "цепочка из двух инструментов"),
    ("L03", "Отбери товары дешевле 20 и посчитай по ним среднее", [BOOKS],
     ["extract_offers", "filter_offers", "compute_stats"], "цепочка из трёх"),
    ("L04", "Переведи цены в рубли", [BOOKS],
     ["extract_offers", "normalize_offers_currency"],
     "конвертация из фунтов: пары GBP в таблице нет"),
    ("L05", "Переведи цены в евро", [SHOP],
     ["extract_offers", "normalize_offers_currency"],
     "конвертация из долларов в евро: пары USD в EUR в таблице нет"),
    ("L06", "Переведи цены в рубли", [SHOP],
     ["extract_offers", "normalize_offers_currency"],
     "конвертация из долларов в рубли: пара есть, должно сработать"),
    ("L07", "Извлеки объявления с этой страницы", [URL404], ["extract_offers"],
     "страница отдаёт 404"),
    ("L08", "Извлеки объявления с этой страницы", [PF], ["extract_offers"],
     "большая страница: видно, сколько доходит до модели"),
    ("L09", "Отбери объявления дешевле 20", [], [],
     "без ссылок: видно, по какому списку работает фильтр"),
    ("L10", "Найди товары дешевле 100 и посчитай среднее", [BOOKS],
     ["extract_offers", "filter_offers", "compute_stats"],
     "граница фильтра, через которую проходят нулевые цены"),
]


def run():
    print("Локальный прогон агента из ai-agent_l5")
    print(f"Модель агента: {cfg.JUDGE_MODEL} через {cfg.PROXY_BASE_URL}\n")

    # Оборачиваем инструменты ОДИН раз: обёртка поверх обёртки писала бы
    # каждый шаг по нескольку раз.
    sink = []
    instrument(A.tools, sink)

    records = []
    for case_id, question, urls, expected_tools, comment in CASES:
        sink.clear()
        # L09 намеренно идёт без ссылок и без очистки состояния: так видно,
        # по какому списку работает фильтр, когда извлечения не было
        if case_id != "L09":
            A._SESSION_STATE["offers"] = []

        print(f"[{case_id}] {comment}")
        print(f"      вопрос: {question}")

        started = time.time()
        error = None
        output = ""
        try:
            result = A.executor.invoke({"question": question, "urls": urls, "max_items": 50})
            output = result.get("output", "")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"      ИСКЛЮЧЕНИЕ: {error[:110]}")
            traceback.print_exc(limit=1)
        elapsed = round(time.time() - started, 2)

        for step in sink:
            mark = "ok " if step["status"] == "success" else "ОШИБКА"
            items = (step.get("metadata") or {}).get("items")
            tail = f", элементов {items}" if items is not None else ""
            print(f"      {mark} {step['name']}({json.dumps(step['input'], ensure_ascii=False)[:70]})"
                  f" {step['duration_ms']} мс{tail}")
            if step["error"]:
                print(f"           {step['error_type']}: {step['error'][:90]}")
        if not sink:
            print("      инструменты не вызывались")
        if output:
            print(f"      ответ: {str(output)[:110]}")
        print(f"      всего {elapsed} с\n")

        records.append({
            "id": case_id,
            "test_type": "local_trace",
            "question": question,
            "urls": urls,
            "expected_tools": expected_tools,
            "comment": comment,
            "output": output,
            "tools_used": [s["name"] for s in sink],
            "error": error,
            "latency_sec": elapsed,
            "trace": list(sink),
        })
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    steps = [s for r in records for s in r["trace"]]
    failed_steps = [s for s in steps if s["status"] == "error"]
    print(f"Кейсов: {len(records)}, шагов в трейсах: {len(steps)}, "
          f"из них упавших: {len(failed_steps)}")
    print(f"Шагов с аргументами: {sum(1 for s in steps if s['input'] is not None)}")
    print(f"Шагов с результатом: {sum(1 for s in steps if s['output'] is not None)}")
    if failed_steps:
        print("\nУпавшие шаги с причиной:")
        for s in failed_steps:
            print(f"  {s['name']}: {s['error_type']}: {s['error'][:80]}")
    print(f"\nТрейсы сохранены в {OUT}")


if __name__ == "__main__":
    run()
