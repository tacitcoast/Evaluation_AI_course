import json
import os

import pandas as pd

from homework_data import PROMPTS
from ai_client import openai_chat_v3, google_chat_v3
from reference_free import (
    calculate_perplexity,
    calculate_readability,
    calculate_coherence_with_transformers,
    calculate_self_bleu,
    calculate_distinct,
)
from reference_based import (
    calculate_rouge,
    calculate_meteor,
    calculate_sacrebleu,
    calculate_exact_match,
)
from embedding_metrics import calculate_bertscore, calculate_semantic_similarity

MODELS = {
    "gpt-5.4-nano": lambda p: openai_chat_v3(p, "gpt-5.4-nano", 0.1),
    "gemini-3-flash-preview": lambda p: google_chat_v3(p, "gemini-3-flash-preview", 0.1),
}

ANSWERS_FILE = "answers.json"
TEMPERATURE = 0.1


def generate_answers():
    if os.path.exists(ANSWERS_FILE):
        print(f"Answers loaded from {ANSWERS_FILE}, no API calls made")
        return json.load(open(ANSWERS_FILE, encoding="utf-8"))

    answers = {}
    for name, call in MODELS.items():
        answers[name] = []
        for item in PROMPTS:
            print(f"  {name} <- prompt {item['id']} ({item['type']})")
            answers[name].append(call(item["prompt"]).strip())

    json.dump(answers, open(ANSWERS_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Answers saved to {ANSWERS_FILE}")
    return answers


def reference_free_metrics(texts):
    perplexity = []
    readability = []
    coherence = []
    self_bleu = []

    for i, text in enumerate(texts):
        perplexity.append(calculate_perplexity(text))
        readability.append(calculate_readability(text)["flesch"])
        first_order = calculate_coherence_with_transformers(text)["first_order"]
        if first_order is not None:
            coherence.append(first_order)
        self_bleu.append(calculate_self_bleu(i, texts))

    distinct = calculate_distinct(texts)

    return {
        "Perplexity (lower = better)": sum(perplexity) / len(perplexity),
        "Flesch readability (higher = easier)": sum(readability) / len(readability),
        "Coherence (higher = better)": sum(coherence) / len(coherence) if coherence else None,
        "Self-BLEU (lower = more diverse)": sum(self_bleu) / len(self_bleu),
        "Distinct-1 (higher = more diverse)": distinct["distinct_1"],
        "Distinct-2 (higher = more diverse)": distinct["distinct_2"],
    }


def reference_based_metrics(predictions, references):
    rouge = calculate_rouge(predictions, references)
    meteor = calculate_meteor(predictions, references)
    sacrebleu = calculate_sacrebleu(predictions, references)
    exact = calculate_exact_match(predictions, references)

    return {
        "ROUGE-1": rouge["overall"]["rouge1"],
        "ROUGE-L": rouge["overall"]["rougeL"],
        "METEOR": meteor["overall"],
        "SacreBLEU": sacrebleu["overall"],
        "Exact Match": exact["overall"],
    }


def embedding_metrics(predictions, references):
    bert_f1 = []
    similarity = []

    for prediction, reference in zip(predictions, references):
        bert_f1.append(calculate_bertscore([reference, prediction])["overall"]["f1"])
        similarity.append(
            calculate_semantic_similarity([reference, prediction])["overall_similarity_mean"]
        )

    return {
        "BERTScore F1": sum(bert_f1) / len(bert_f1),
        "Sentence-BERT similarity (LaBSE)": sum(similarity) / len(similarity),
        "_per_prompt_similarity": similarity,
        "_per_prompt_bertscore": bert_f1,
    }


def main():
    print("=" * 70)
    print("AUTOMATIC LLM EVALUATION")
    print("=" * 70)
    print(f"Prompts: {len(PROMPTS)}, models: {len(MODELS)}, temperature: {TEMPERATURE}")

    print("")
    print("1. Generating answers")
    answers = generate_answers()

    references = [item["reference"] for item in PROMPTS]

    print("")
    print("=" * 70)
    print("ANSWERS")
    print("=" * 70)
    for item in PROMPTS:
        print("")
        print(f"[{item['id']}] {item['type']}: {item['prompt'][:70]}")
        print(f"  reference: {item['reference'][:90]}")
        for name in MODELS:
            print(f"  {name}: {answers[name][item['id'] - 1][:90]}")

    free, based, embed = {}, {}, {}
    for name in MODELS:
        print("")
        print(f"2. Metrics for {name}")
        predictions = answers[name]
        free[name] = reference_free_metrics(predictions)
        based[name] = reference_based_metrics(predictions, references)
        embed[name] = embedding_metrics(predictions, references)

    for title, data in [("REFERENCE-FREE METRICS", free),
                        ("REFERENCE-BASED METRICS", based),
                        ("EMBEDDING-BASED METRICS", embed)]:
        table = pd.DataFrame({
            name: {k: v for k, v in values.items() if not k.startswith("_")}
            for name, values in data.items()
        })
        print("")
        print("=" * 70)
        print(title)
        print("=" * 70)
        print(table.to_string(float_format="%.4f"))

    print("")
    print("=" * 70)
    print("PER-PROMPT VIEW (semantic similarity vs exact match)")
    print("=" * 70)
    rows = []
    for item in PROMPTS:
        i = item["id"] - 1
        row = {"id": item["id"], "type": item["type"]}
        for name in MODELS:
            row[f"{name[:12]} sim"] = embed[name]["_per_prompt_similarity"][i]
            row[f"{name[:12]} match"] = int(
                answers[name][i].strip().lower() == references[i].strip().lower())
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))

    json.dump({"reference_free": free, "reference_based": based,
               "embedding_based": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                   for k, v in embed.items()}},
              open("metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("")
    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    main()
