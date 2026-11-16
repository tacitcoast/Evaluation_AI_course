import json
import os
import time

import pandas as pd
import requests

ENDPOINT = "https://rag.eval-ai.com/api/v1/chat/"
API_KEY = "80456142-5441-4469-b97f-1d72b7802a93"
FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(FOLDER, "data")
OUT = os.path.join(FOLDER, "data", "homework_rag_answers.json")

DATASETS = {
    "happy_path": "homework_happy_path.csv",
    "negative": "homework_negative.csv",
    "metamorphic": "homework_metamorphic.csv",
}


def ask(question):
    response = requests.post(
        ENDPOINT,
        json={"message": question},
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def main():
    collected = {}
    if os.path.exists(OUT):
        collected = json.load(open(OUT, encoding="utf-8"))

    for name, filename in DATASETS.items():
        df = pd.read_csv(os.path.join(DATA, filename))
        collected.setdefault(name, [])
        done = {row["question"] for row in collected[name]}

        for i, row in df.iterrows():
            question = row["question"]
            if question in done:
                continue

            try:
                data = ask(question)
                item = {
                    "question": question,
                    "expected_response": row["expected_response"],
                    "answer": data.get("content", ""),
                    "contexts": [s.get("content", "") for s in data.get("sources", [])],
                    "sources_meta": [
                        {"document": s.get("document_title"), "similarity": s.get("similarity")}
                        for s in data.get("sources", [])
                    ],
                    "error": None,
                }
            except Exception as exc:
                item = {
                    "question": question,
                    "expected_response": row["expected_response"],
                    "answer": "",
                    "contexts": [],
                    "sources_meta": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }

            for extra in ("test_type", "based_on", "transformation_type"):
                if extra in df.columns:
                    item[extra] = row[extra]

            collected[name].append(item)
            json.dump(collected, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

            status = "ERROR" if item["error"] else f"{len(item['contexts'])} chunks"
            print(f"[{name}] {i + 1}/{len(df)} {status}: {question[:55]}", flush=True)
            time.sleep(0.3)

    print("")
    print("COLLECTED")
    for name, rows in collected.items():
        failed = sum(1 for r in rows if r["error"])
        print(f"  {name:<12} {len(rows):>3} answers, errors: {failed}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
