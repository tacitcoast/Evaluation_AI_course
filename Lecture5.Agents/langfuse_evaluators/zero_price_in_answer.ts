// Langfuse Cloud, Code evaluator "zero_price_in_answer".
// Метрика №1 онлайн-оценки. Исходник хранится здесь, потому что в Langfuse
// код живёт только внутри UI эвалуатора и в репозиторий иначе не попадает.
//
// Что ловит: агент иногда отдаёт "0 AED" / "0 руб" как настоящую цену —
// это не ошибка парсинга цен на странице, а молчаливая подмена данных нулём.
// 1 — нулевых цен нет, 0 — ноль подан как цена.
function evaluate(ctx: EvaluationContext): EvaluationResult {
  const raw = ctx.observation.output;
  const text = typeof raw === "string" ? raw : JSON.stringify(raw ?? "");
  const cur = "zł|PLN|AED|USD|EUR|RUB|GBP|руб|₽|\\$|£|€";
  const re = new RegExp("(^|[^0-9.,])0+([.,]0+)?\\s*(" + cur + ")", "gi");
  const hits = text.match(re) || [];
  const clean = hits.length === 0;
  return {
    scores: [{
      name: "zero_price_in_answer",
      value: clean ? 1 : 0,
      dataType: "NUMERIC",
      comment: clean
        ? "Нулевых цен в ответе нет"
        : "Ноль подан как цена, совпадений: " + hits.length + " — " + hits.slice(0, 5).join(", "),
    }],
  };
}
