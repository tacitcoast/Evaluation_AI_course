# Evaluation AI Course

Учебный репозиторий курса о том, как измерять качество AI-систем. Здесь материалы лекций и мои домашние работы.

## Программа

| Лекция | О чём |
|---|---|
| [1. Python](Lecture1.Python) | основы Python для работы с данными; метрики считаем руками, без библиотек |
| [2. ML/DL](Lecture2.ML_DL) | accuracy, precision, recall, F1, MAE, RMSE, R²; матрица ошибок, переобучение, калибровка |
| [3. LLM](Lecture3.LLM) | метрики с эталоном (ROUGE, BLEU, METEOR) и без него; LLM-as-a-Judge, бенчмарки |
| [4. RAG](Lecture4.RAG) | своя RAG-система на FastAPI и ChromaDB, оценка через DeepEval, RAGAS, Opik, promptfoo |
| [5. Agents](Lecture5.Agents) | агент на LangChain; ToolCorrectness и TaskCompletion — важен путь, а не только ответ |
| [6. Multiturn & Multimodal](Lecture6.Multiturn_and_Miltimodal) | оценка диалога целиком; CLIP Score и LPIPS для картинок |

Сквозная мысль всего курса: одна общая цифра качества почти всегда врёт. Что бы ни измерялось — спам, удовлетворённость пассажира, ответ бота — ошибки надо разбирать по типам.

## Как устроен репозиторий

Каждая лекция — самостоятельная папка: материалы преподавателя, мои работы `homework*`, свой `requirements.txt` и README с заданием, решением и выводами.

Между лекциями код не переиспользуется, окружение заводится отдельно под каждую:

```
cd Lecture2.ML_DL && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Инструменты

Фреймворки оценки: DeepEval, RAGAS, Opik, promptfoo.
Обучение и трекинг: PyTorch, Keras, scikit-learn, TensorBoard, Weights & Biases.
Наблюдаемость агентов: LangSmith, Langfuse.

## Куда смотреть дальше

Открытые проекты, по которым полезно сверяться с индустрией:

- [OpenAI Evals](https://github.com/openai/evals) — фреймворк и реестр тестов для LLM;
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — стандарт для прогона академических бенчмарков;
- [HELM](https://github.com/stanford-crfm/helm) — оценка сразу по многим измерениям, не только по точности;
- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) — фреймворк оценок от UK AI Safety Institute;
- [MTEB](https://github.com/embeddings-benchmark/mteb) — бенчмарк моделей эмбеддингов;
- [Giskard](https://github.com/Giskard-AI/giskard) — поиск уязвимостей и нежелательного поведения;
- [TruLens](https://github.com/truera/trulens) и [Phoenix](https://github.com/Arize-ai/phoenix) — оценка вместе с трассировкой запросов.
