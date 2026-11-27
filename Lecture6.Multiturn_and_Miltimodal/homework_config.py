# homework_config.py
"""Настройки домашки по лекции 6: читаются из Lecture6.Multiturn_and_Miltimodal/.env

Ключи в коде не держим — файл .env в .gitignore.
Образец значений лежит в .env.example.
"""

import os
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"


def _load_env(path: Path = ENV_PATH) -> None:
    """Простейший разбор .env, чтобы не тянуть python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

# --- агент под тестом (мультитурн-вариант, лекция 6) --------------------
AGENT_URL = os.getenv("AGENT_URL", "http://5.11.83.110:8005/ask")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")
AGENT_USER_ID = os.getenv("AGENT_USER_ID", "tacitcoast")

# --- модель-судья через курсовой прокси ---------------------------------
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "http://5.11.83.110:8000")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

# --- пороги и объёмы прогона --------------------------------------------
# Порог зафиксирован до прогона и в коде не меняется по результатам.
METRIC_THRESHOLD = float(os.getenv("METRIC_THRESHOLD", "0.5"))

# --- пути ----------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DIALOGS_PATH = DATA_DIR / "homework_dialogs.json"
MULTITURN_RESULTS_PATH = DATA_DIR / "homework_multiturn_results.json"
MULTIMODAL_RESULTS_PATH = DATA_DIR / "homework_multimodal_results.json"
IMAGES_DIR = BASE_DIR / "multimodal_eval" / "sample_images"


def require(name: str) -> str:
    """Возвращает значение переменной или объясняет, чего не хватает."""
    value = globals().get(name) or os.getenv(name, "")
    if not value:
        raise SystemExit(
            f"Не задано {name}. Скопируйте .env.example в .env и заполните значения."
        )
    return value
