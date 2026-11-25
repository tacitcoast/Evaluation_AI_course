// Langfuse Cloud, Code evaluator "price_stats_consistency".
// Метрика №2 онлайн-оценки.
//
// Что ловит: агент считает мин/макс/среднее/медиану по ценам и иногда
// возвращает внутренне противоречивый набор (среднее вне диапазона,
// медиана вне диапазона, минимум больше максимума, минимум равен нулю).
// Проверка не требует LLM — достаточно арифметики, поэтому это Code-метрика.
// Если в ответе меньше двух величин, проверять нечего: ставим 1 и говорим об этом.
function evaluate(ctx: EvaluationContext): EvaluationResult {
  const raw = ctx.observation.output;
  const text = typeof raw === "string" ? raw : JSON.stringify(raw ?? "");
  const grab = (label: string) =>
    ((text.match(new RegExp(label + "[^0-9]{0,40}([0-9][0-9 .,]*)", "i")) || ["", ""])[1] || "")
      .replace(/[^0-9.,]/g, "")
      .replace(/,(\d\d?)$/, ".$1")
      .replace(/,/g, "");
  const toNum = (s: string) => (s === "" || isNaN(parseFloat(s)) ? null : parseFloat(s));
  const min = toNum(grab("минимальн"));
  const max = toNum(grab("максимальн"));
  const avg = toNum(grab("средн"));
  const med = toNum(grab("медиан"));
  const known = [min, max, avg, med].filter(v => v !== null);
  const bad: string[] = [];
  if (min !== null && max !== null && min > max) bad.push("минимум больше максимума");
  if (avg !== null && min !== null && avg < min) bad.push("среднее меньше минимума");
  if (avg !== null && max !== null && avg > max) bad.push("среднее больше максимума");
  if (med !== null && min !== null && med < min) bad.push("медиана меньше минимума");
  if (med !== null && max !== null && med > max) bad.push("медиана больше максимума");
  if (min === 0) bad.push("минимальная цена равна нулю");
  const skip = known.length < 2;
  const ok = bad.length === 0;
  return {
    scores: [{
      name: "price_stats_consistency",
      value: skip || ok ? 1 : 0,
      dataType: "NUMERIC",
      comment: skip
        ? "Статистики в ответе нет, проверять нечего"
        : ok
        ? "Статистика непротиворечива, величин: " + known.length
        : bad.join("; "),
    }],
  };
}
