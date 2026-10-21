import os
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
from google import genai
from google.genai import types

from benchmarks import form_options

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

CATEGORY = "computer science"
N = 2


def build_prompts():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")

    cot_prefix = ""
    for d in dataset["validation"]:
        if d["category"] == CATEGORY:
            cot_prefix += "Q: " + d["question"] + "\n" + form_options(d["options"]) + "\n" + d["cot_content"] + "\n\n"

    test_set = [ex for ex in dataset["test"] if ex["category"] == CATEGORY][:N]

    prompts = []
    for example in test_set:
        block = "Q: " + example["question"] + "\n" + form_options(example["options"]) + "\n"
        prompts.append(cot_prefix + block)
    return cot_prefix, prompts


def main():
    cot_prefix, prompts = build_prompts()

    print(f"Few-shot prefix: {len(cot_prefix):,} characters")
    print(f"Full prompt:     {len(prompts[0]):,} characters")
    print("")

    totals = {"gpt-5.4-nano": [0, 0], "gemini-3-flash-preview": [0, 0]}

    for i, prompt in enumerate(prompts, 1):
        r = openai_client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        totals["gpt-5.4-nano"][0] += r.usage.prompt_tokens
        totals["gpt-5.4-nano"][1] += r.usage.completion_tokens
        print(f"Q{i} gpt-5.4-nano:           in {r.usage.prompt_tokens:>6}  out {r.usage.completion_tokens:>5}")

        g = google_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        u = g.usage_metadata
        totals["gemini-3-flash-preview"][0] += u.prompt_token_count
        totals["gemini-3-flash-preview"][1] += (u.candidates_token_count or 0)
        print(f"Q{i} gemini-3-flash-preview: in {u.prompt_token_count:>6}  out {u.candidates_token_count or 0:>5}")

    print("")
    print("=" * 64)
    print("AVERAGE PER CALL")
    print("=" * 64)
    grand = 0
    for name, (tin, tout) in totals.items():
        avg = (tin + tout) / N
        grand += avg
        print(f"{name:<24} in {tin // N:>6}  out {tout // N:>5}  total {avg:>8,.0f}")

    print("")
    print("=" * 64)
    print("FORECAST FOR THE FULL RUN")
    print("=" * 64)
    print("16 benchmark runs per model; MMLU-Pro is the most expensive one,")
    print("so this is an upper bound, not an exact number.")
    print("")
    for n in (5, 10, 20):
        calls = 16 * 2 * n
        print(f"  {n:>2} questions: {calls:>4} calls, up to {grand / 2 * calls:>12,.0f} tokens")


if __name__ == "__main__":
    main()
