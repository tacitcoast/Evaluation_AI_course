# Лекция 6. Многоходовые диалоги и мультимодальность

Два независимых блока. Многоходовая оценка смотрит на беседу целиком:
кроме отдельного ответа появляются удержание контекста, удержание роли и
доведение задачи до результата. Мультимодальная оценка добавляет к тексту
изображение, и метрики здесь разного рода — пиксельные, эмбеддинговые и на
модели-судье.

## Материалы лекции

| Файл | О чём |
|---|---|
| `ai-agent_l7/` | агент недвижимости, мультитурн-вариант, `POST /ask` на порту 8005 |
| `ai-agent_l7/app/agent/agent.py` | инструменты, системный промпт и состояние сессии |
| `multiturn_eval/chat_cli.py` | интерактивный CLI: запись беседы и запуск оценки |
| `multiturn_eval/agent_connector.py` | клиент к `/ask`, нормализует ответ в `{output, tools_used}` |
| `multiturn_eval/conversation_storage.py` | хранение бесед в JSON |
| `multiturn_eval/evaluators/` | один и тот же прогон на DeepEval и Opik |
| `multimodal_eval/metrics/clip_score.py` | CLIP Score: изображение–текст и изображение–изображение |
| `multimodal_eval/metrics/lpips_metric.py` | LPIPS: перцептуальное расстояние |
| `multimodal_eval/metrics/deepeval_multimodal.py` | ImageCoherence, ImageHelpfulness, ImageReference, TextToImage |
| `multimodal_eval/sample_images/` | три снимка: `original`, `edited`, `different` |

## Мои работы

| Файл | Задание |
|---|---|
| `homework_config.py` | ключи и адреса из `.env`, образец в `.env.example` |
| `homework_judge.py` | судья через прокси со строгим JSON по pydantic-схеме |
| `homework_dialogs.py` | 1 — шесть сценариев бесед с записанным признаком провала |
| `homework_collect_dialogs.py` | 1 — запись бесед, стадия сбора |
| `homework_evaluate_multiturn.py` | 1 — четыре многоходовые метрики, стадия оценки |
| `homework_analyze.py` | 1 — сведение и сопоставление метрик с чтением |
| `data/homework_dialogs.json` | 1 — записанные беседы |
| `data/homework_multiturn_results.json`, `homework_multiturn_summary.json` | 1 — результаты и сводка |
| `Multiturn_Evaluation_Report.docx`, `.md` | 1 — сдаваемый отчёт |
| `homework_multimodal_images.py` | 2 — подготовка группы контролируемых искажений |
| `homework_multimodal.py` | 2 — четыре группы примеров, CLIP, LPIPS, метрики DeepEval |
| `data/homework_multimodal_results.json` | 2 — результаты |
| `Multimodal_Evaluation_Report.docx`, `.md` | 2 — сдаваемый отчёт |

## Как запускать

Своё окружение на Python 3.13: системный 3.9 для DeepEval 4.x не подходит.

```bash
cd Lecture6.Multiturn_and_Miltimodal
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install 'deepeval==4.1.8' opik torch torchvision \
    torchmetrics lpips 'transformers<5' Pillow python-dotenv openai \
    requests pandas numpy openpyxl markdown
cp .env.example .env   # и заполнить
```

Практика 1, три стадии подряд:

```bash
.venv/bin/python homework_collect_dialogs.py     # запись бесед
.venv/bin/python homework_evaluate_multiturn.py  # метрики
.venv/bin/python homework_analyze.py             # сведение
```

Практика 2:

```bash
.venv/bin/python homework_multimodal_images.py   # группа искажений
.venv/bin/python homework_multimodal.py          # все четыре группы
```

Обе стадии практики 1 дописывают состояние после каждого шага: повторный
запуск продолжает с места обрыва, уже сделанное не пересчитывается.

## Что нашлось

Практика 1. Агент не хранит сказанное пользователем: состояние сессии
содержит только извлечённые объявления, истории сообщений нет. На первом ходу
агент отвечает «Запомнил ваш бюджет — 800 тысяч дирхам», на третьем — «у вас
пока нет данных о бюджете». Ручное чтение дало провал по всем шести беседам,
метрики согласились полностью ни по одной. `KnowledgeRetention` не опустилась
ниже порога ни разу при четырёх явных потерях контекста из шести бесед.

Практика 2. CLIP image-image уложил разницу между «тот же снимок» и «другой
человек» в пять пунктов из ста, а сильно размытую копию того же снимка оценил
ниже, чем снимок другого человека. Подменённая подпись отличается от верной на
2–3 пункта, посторонняя — на 17. Метрики DeepEval на согласованность,
полезность и ссылку дали одинаковые оценки для соответствующего и
несоответствующего снимка.

Подробности и числа — в отчётах.
