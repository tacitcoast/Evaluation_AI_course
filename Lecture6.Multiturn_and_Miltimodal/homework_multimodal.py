# homework_multimodal.py
"""Практика 2: оценка изображений на трёх группах примеров.

Группы:
  1. Пары снимков — тот же сюжет против другого. CLIP image-image, LPIPS.
  2. Изображение и подпись — верная, подменённая, посторонняя. CLIP image-text.
  3. Контролируемые искажения одного снимка нарастающей силы. CLIP, LPIPS.
  4. Изображение в тексте — мультимодальные метрики DeepEval на MLLM-судье.

Смысл набора не в том, чтобы получить числа, а в том, чтобы проверить, на что
метрики реагируют: группа 3 с известной силой искажения показывает это прямо.

Запуск:
    .venv/bin/python homework_multimodal.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "multimodal_eval"))

import homework_config as cfg
from metrics.clip_score import (
    compute_clip_score_text,
    compute_clip_score_images,
    interpret_clip_score,
)
from metrics.lpips_metric import compute_lpips, interpret_lpips

IMAGES = cfg.IMAGES_DIR
DERIVED = IMAGES / "derived"

# Подписи для группы 2. Текст CLIP понимает по-английски заметно лучше,
# поэтому подписи на английском; русские варианты добавлены для сравнения.
CAPTIONS = {
    "верная для original":
        "a man with glasses working on a laptop with code on two monitors",
    "верная для different":
        "a man in a beige sweater at a desk with charts and graphs on a monitor",
    "посторонняя":
        "a cat sleeping on a sofa in a living room",
    "верная, но по-русски":
        "мужчина в очках работает за ноутбуком, на мониторах код",
}


def group_1_pairs() -> list:
    """Пары снимков: насколько метрики различают «тот же сюжет» и «другой»."""
    pairs = [
        ("original", "original", "одно и то же изображение"),
        ("original", "edited", "тот же человек, другой кадр"),
        ("original", "different", "другой человек, другой монитор"),
        ("edited", "different", "другой человек, другой монитор"),
    ]
    rows = []
    for left, right, note in pairs:
        a, b = IMAGES / f"{left}.png", IMAGES / f"{right}.png"
        clip = compute_clip_score_images(a, b)
        lp = compute_lpips(a, b)
        rows.append({
            "group": 1, "left": left, "right": right, "note": note,
            "clip_image_image": round(clip, 2),
            "lpips": round(lp, 4),
            "lpips_reading": interpret_lpips(lp),
        })
        print(f"  {left:>10} ↔ {right:<10} CLIP {clip:6.2f}   LPIPS {lp:.4f}  {note}")
    return rows


def group_2_captions() -> list:
    """Изображение и подпись: отличает ли CLIP верную подпись от подменённой."""
    rows = []
    for image_name in ("original", "different"):
        path = IMAGES / f"{image_name}.png"
        print(f"  {image_name}.png")
        for label, text in CAPTIONS.items():
            score = compute_clip_score_text(path, text)
            rows.append({
                "group": 2, "image": image_name, "caption_kind": label,
                "caption": text,
                "clip_image_text": round(score, 2),
                "reading": interpret_clip_score(score),
            })
            print(f"     {label:<24} {score:6.2f}  {interpret_clip_score(score)}")
    return rows


def group_3_distortions() -> list:
    """Контролируемые искажения: монотонны ли метрики по силе искажения."""
    source = IMAGES / "original.png"
    order = [
        ("jpeg_low", "сжатие JPEG, качество 10"),
        ("noise", "гауссов шум, сигма 25"),
        ("blur_light", "размытие, радиус 2"),
        ("blur_heavy", "размытие, радиус 8"),
        ("grayscale", "оттенки серого"),
        ("mirror", "отражение по горизонтали"),
    ]
    rows = []
    for name, description in order:
        path = DERIVED / f"{name}.png"
        clip = compute_clip_score_images(source, path)
        lp = compute_lpips(source, path)
        caption_score = compute_clip_score_text(path, CAPTIONS["верная для original"])
        rows.append({
            "group": 3, "transform": name, "description": description,
            "clip_image_image": round(clip, 2),
            "lpips": round(lp, 4),
            "clip_image_text": round(caption_score, 2),
        })
        print(f"  {name:<12} CLIP↔ {clip:6.2f}   LPIPS {lp:.4f}   "
              f"CLIP-подпись {caption_score:6.2f}   {description}")
    return rows


def group_4_mllm() -> list:
    """Изображение внутри текста: мультимодальные метрики DeepEval."""
    from metrics.deepeval_multimodal import (
        evaluate_image_coherence,
        evaluate_image_helpfulness,
        evaluate_image_reference,
        evaluate_text_to_image,
    )

    text_matching = [
        "Разработчик за рабочим местом: ноутбук и два внешних монитора с кодом.",
        str(IMAGES / "original.png"),
        "На снимке видно, что весь код открыт на внешних мониторах, "
        "а ноутбук используется как клавиатура.",
    ]
    text_mismatching = [
        "Разработчик за рабочим местом: ноутбук и два внешних монитора с кодом.",
        str(IMAGES / "different.png"),
        "На снимке видно, что весь код открыт на внешних мониторах, "
        "а ноутбук используется как клавиатура.",
    ]

    cases = [
        ("согласованный текст и снимок", text_matching),
        ("текст про код, снимок с графиками", text_mismatching),
    ]
    checks = [
        ("ImageCoherence", evaluate_image_coherence),
        ("ImageHelpfulness", evaluate_image_helpfulness),
        ("ImageReference", evaluate_image_reference),
    ]

    rows = []
    prompt = "Опиши рабочее место разработчика и поясни, как оно устроено."
    for case_name, output in cases:
        print(f"  {case_name}")
        for metric_name, func in checks:
            started = time.time()
            try:
                result = func(prompt, output,
                              threshold=cfg.METRIC_THRESHOLD, model=cfg.JUDGE_MODEL)
                rows.append({"group": 4, "case": case_name, "metric": metric_name,
                             "score": result["score"], "passed": result["success"],
                             "reason": result["reason"], "error": None})
                print(f"     {metric_name:<18} {result['score']:.2f}  "
                      f"{'прошла' if result['success'] else 'НЕ ПРОШЛА'}"
                      f"  ({round(time.time() - started, 1)} с)")
            except Exception as exc:
                rows.append({"group": 4, "case": case_name, "metric": metric_name,
                             "score": None, "passed": None, "reason": None,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"     {metric_name:<18} ОШИБКА: {str(exc)[:80]}")

    # TextToImage: снимок против запроса, которому он соответствует и не соответствует
    for label, prompt_text, image_name in [
        ("запрос совпадает со снимком",
         "фотография разработчика в очках за ноутбуком, на мониторах код",
         "original"),
        ("запрос не совпадает со снимком",
         "фотография пустого пляжа на закате",
         "original"),
    ]:
        started = time.time()
        try:
            result = evaluate_text_to_image(
                prompt_text, IMAGES / f"{image_name}.png",
                threshold=cfg.METRIC_THRESHOLD, model=cfg.JUDGE_MODEL)
            rows.append({"group": 4, "case": label, "metric": "TextToImage",
                         "score": result["score"], "passed": result["success"],
                         "reason": result["reason"], "error": None})
            print(f"  {label:<34} TextToImage {result['score']:.2f}  "
                  f"({round(time.time() - started, 1)} с)")
        except Exception as exc:
            rows.append({"group": 4, "case": label, "metric": "TextToImage",
                         "score": None, "passed": None, "reason": None,
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {label:<34} TextToImage ОШИБКА: {str(exc)[:80]}")
    return rows


def main() -> int:
    if not (DERIVED / "blur_light.png").exists():
        raise SystemExit("Нет производных изображений. Сначала homework_multimodal_images.py")

    rows = []
    print("\nГруппа 1. Пары снимков")
    rows += group_1_pairs()
    print("\nГруппа 2. Изображение и подпись")
    rows += group_2_captions()
    print("\nГруппа 3. Контролируемые искажения")
    rows += group_3_distortions()
    print("\nГруппа 4. Изображение внутри текста, судья на MLLM")
    rows += group_4_mllm()

    cfg.DATA_DIR.mkdir(exist_ok=True)
    cfg.MULTIMODAL_RESULTS_PATH.write_text(
        json.dumps({"finished_at": datetime.now().isoformat(timespec="seconds"),
                    "threshold": cfg.METRIC_THRESHOLD,
                    "judge": cfg.JUDGE_MODEL,
                    "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nЗаписей: {len(rows)}")
    print(f"Файл: {cfg.MULTIMODAL_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
