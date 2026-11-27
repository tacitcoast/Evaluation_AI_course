# homework_judge.py
"""Судья на языковой модели через курсовой прокси, со строгим JSON.

Локальный ProxyLLM лекции (multiturn_eval/evaluators/deepeval/deepeval_custom_llm.py)
умеет только свободный текст: DeepEval затем разбирает его вручную и падает на
кривом выводе. Здесь судья возвращает объект pydantic-схемы, которую запросил
DeepEval, с фолбэком на JSON-mode для прокси без structured output.
"""

from typing import Optional, Type

from openai import OpenAI
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM

import homework_config as cfg


class ProxyJudge(DeepEvalBaseLLM):
    """Модель-судья, обращающаяся к прокси курса."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self._model_name = model or cfg.JUDGE_MODEL
        self.temperature = temperature
        self.api_key = api_key or cfg.require("PROXY_API_KEY")
        self.base_url = base_url or cfg.PROXY_BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = self.client
        # Счётчик обращений: судья на каждой реплике — это деньги.
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def load_model(self):
        return self.client

    def _account(self, response) -> None:
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        if schema is not None:
            return self._generate_with_schema(prompt, schema)
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        self._account(response)
        return response.choices[0].message.content

    def _generate_with_schema(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        try:
            response = self.client.beta.chat.completions.parse(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=schema,
            )
            self._account(response)
            return response.choices[0].message.parsed
        except Exception:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            self._account(response)
            return schema.model_validate_json(response.choices[0].message.content)

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self._model_name
