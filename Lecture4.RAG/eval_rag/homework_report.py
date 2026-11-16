import json
import os
from datetime import date

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(FOLDER, "data")
OUT = os.path.join(os.path.dirname(FOLDER), "RAG_Evaluation_Report.docx")

DARK = RGBColor(0x1F, 0x38, 0x64)
TEXT = RGBColor(0x26, 0x26, 0x26)


def load():
    ev = json.load(open(os.path.join(DATA, "homework_results_evaliqa.json"), encoding="utf-8"))
    de = json.load(open(os.path.join(DATA, "homework_results_deepeval.json"), encoding="utf-8"))
    fx = json.load(open(os.path.join(DATA, "homework_faithfulness_fixed.json"), encoding="utf-8"))
    ans = json.load(open(os.path.join(DATA, "homework_rag_answers.json"), encoding="utf-8"))

    de_means, de_below = {}, {}
    for item in de:
        for name, v in item["scores"].items():
            if v["score"] is not None:
                de_means.setdefault(name, []).append(v["score"])
    for name, vals in de_means.items():
        de_below[name] = sum(1 for v in vals if v < 0.7)
        de_means[name] = sum(vals) / len(vals)
    return ev, de_means, de_below, fx, ans


def title(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(20)
    r.font.bold = True
    r.font.color.rgb = DARK
    return p


def heading(doc, text, size=14):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = DARK
    return p


def body(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(11)
    r.font.color.rgb = TEXT
    return p


def bullet(doc, text, style="List Number"):
    p = doc.add_paragraph(style=style)
    r = p.add_run(text)
    r.font.size = Pt(11)
    r.font.color.rgb = TEXT
    return p


def table(doc, rows, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(value))
            r.font.size = Pt(9.5)
            r.font.color.rgb = TEXT
            r.font.bold = (i == 0)
    doc.add_paragraph()
    return t


def main():
    ev, de, de_below, fx, ans = load()
    doc = Document()

    title(doc, "RAG Evaluation Report")
    heading(doc, "Report Information", 13)
    table(doc, [
        ["Field", "Your entry"],
        ["Student", "Anna Malinovskaia"],
        ["Lecture", "4. RAG — practice 2"],
        ["Date", date.today().strftime("%d.%m.%Y")],
        ["System under test", "https://rag.eval-ai.com/api/v1/chat/"],
        ["Frameworks", "EvaliQA (eval-ai-library 0.8.4), DeepEval 4.1.8"],
        ["Judge model", "gpt-4.1-mini, threshold 0.7"],
    ])

    heading(doc, "1. Objective")
    body(doc, "I measured the quality of the course RAG system with two frameworks. On happy path "
              "data I ran Answer Relevancy, Context Relevancy and Faithfulness. On negative and "
              "metamorphic data I ran Answer Precision and Restricted Refusal, which exist in "
              "EvaliQA only. The same judge model and the same threshold were used everywhere, so "
              "the two frameworks can be compared directly.")

    heading(doc, "2. Datasets and Setup")
    body(doc, "I wrote three datasets from the 17 requirement documents the system is built on. "
              "The documents describe an online store: microservices, authorization, UI.")
    table(doc, [
        ["Dataset", "Size", "Content"],
        ["Happy path", "50", "35 factual, 11 explanatory, 3 multi-hop, 1 comparative"],
        ["Negative", "20", "14 out of scope, 6 with a false premise"],
        ["Metamorphic", "15", "3 groups, 6 transformation types"],
    ])
    body(doc, "Collecting answers and scoring them are two separate steps. All 85 questions were "
              "sent to the RAG system once and saved to a file with their retrieval contexts. "
              "Metrics were then computed from that file, so a failure in the middle of a run "
              "does not lose the answers.")

    heading(doc, "3. Results")
    body(doc, "Happy path, 50 cases, both frameworks:")
    table(doc, [
        ["Metric", "EvaliQA", "DeepEval", "Below 0.7 (EvaliQA / DeepEval)"],
        ["Answer Relevancy", f"{ev['happy_path']['means']['answerRelevancyMetric']:.3f}",
         f"{de['AnswerRelevancyMetric']:.3f}",
         f"{ev['happy_path']['below_threshold']['answerRelevancyMetric']} / {de_below['AnswerRelevancyMetric']}"],
        ["Faithfulness", f"{ev['happy_path']['means']['faithfulnessMetric']:.3f}",
         f"{de['FaithfulnessMetric']:.3f}",
         f"{ev['happy_path']['below_threshold']['faithfulnessMetric']} / {de_below['FaithfulnessMetric']}"],
        ["Faithfulness (after fix)", f"{fx['mean']:.3f}", f"{de['FaithfulnessMetric']:.3f}",
         f"{fx['below_threshold']} / {de_below['FaithfulnessMetric']}"],
        ["Context Relevancy", f"{ev['happy_path']['means']['contextualRelevancyMetric']:.3f}",
         f"{de['ContextualRelevancyMetric']:.3f}",
         f"{ev['happy_path']['below_threshold']['contextualRelevancyMetric']} / {de_below['ContextualRelevancyMetric']}"],
    ])

    body(doc, "Negative and metamorphic data, EvaliQA:")
    table(doc, [
        ["Dataset", "Metric", "Score", "Below 0.7"],
        ["Negative", "Answer Precision", f"{ev['negative']['means']['answerPrecisionMetric']:.3f}",
         f"{ev['negative']['below_threshold']['answerPrecisionMetric']}/20"],
        ["Negative", "Restricted Refusal", f"{ev['negative']['means']['restrictedRefusalMetric']:.3f}",
         f"{ev['negative']['below_threshold']['restrictedRefusalMetric']}/20"],
        ["Metamorphic", "Answer Precision", f"{ev['metamorphic']['means']['answerPrecisionMetric']:.3f}",
         f"{ev['metamorphic']['below_threshold']['answerPrecisionMetric']}/15"],
    ])

    body(doc, "Context Relevancy by question type (EvaliQA):")
    table(doc, [
        ["Question type", "Cases", "Answer Relevancy", "Context Relevancy"],
        ["factual", "35", "0.950", "0.435"],
        ["explanatory", "11", "0.916", "0.532"],
        ["multi-hop", "3", "0.818", "0.793"],
        ["comparative", "1", "0.943", "0.660"],
    ])

    heading(doc, "4. Errors found in the RAG system")
    body(doc, "Retrieval brings in material that has nothing to do with the question. This is the "
              "main problem and both frameworks agree on it: 39 of 50 cases below the threshold in "
              "EvaliQA, 47 of 50 in DeepEval. For the question about access_token lifetime the "
              "system returned five chunks, and only one was about tokens. The other four were "
              "about email registration and super-admin rights. The answer was still correct, so "
              "the generation step compensates for the weak search.")
    body(doc, "Narrow questions suffer most. Context Relevancy is 0.435 on factual questions and "
              "0.793 on multi-hop ones. The system always returns five chunks regardless of how "
              "specific the question is, so for a one-line question most of the context is noise.")
    body(doc, "The system confirmed a false statement. Asked \"A warehouse with products is deleted "
              "automatically together with the goods, right?\", it answered \"Yes, that is correct\". "
              "The documents say the opposite: a warehouse that holds products cannot be deleted. "
              "This is the only case out of six false-premise questions where the system agreed with "
              "a wrong premise instead of correcting it.")
    body(doc, "On questions outside the documentation the system behaves well. It refused in 10 of "
              "14 such cases and did not invent facts in the rest.")

    heading(doc, "5. Issues found in the evaluation tools")
    body(doc, "Faithfulness in EvaliQA gives 0.00 on Russian answers. The metric checks that the "
              "judge's supporting quote really comes from the context, and the check uses the "
              "regular expression [a-z0-9]{3,}. It matches Latin characters only, so a Russian "
              "quote produces an empty token set and the check returns False. A correct verdict is "
              "then rewritten to \"none\" with weight 0.0. The quote \"Токен должен быть валидным на "
              "протяжении 30 минут\" is present in the context word for word, Python's \"in\" returns "
              "True, and the library's function returns False. Among cases with a non-zero score, "
              "Latin words appear in all 29 of 29 answers; among zero-score cases, in 6 of 21. With "
              "the Cyrillic range added to the regular expression the average rises from 0.528 to "
              "0.926 and matches DeepEval's 0.972.")
    body(doc, "Faithfulness in DeepEval fails on gpt-4o-mini. The judge generates 16384 tokens for a "
              "990-token prompt and hits the output limit. The same metric with gpt-4.1-mini takes "
              "seven seconds per case. All numbers in this report use gpt-4.1-mini.")
    body(doc, "Answer Precision compares texts literally, so its score cannot be read as answer "
              "quality. The system answers correctly but at length, and every extra word lowers the "
              "score. The spread inside metamorphic groups is small — 0.150 to 0.170 in the order "
              "status group — which means the answers barely change when the question is reworded. "
              "The metamorphic tests show that the system is stable. The 0.352 average says more about wording than about correctness.")
    body(doc, "Restricted Refusal only applies to part of the negative set. It scores 0.714 on the "
              "14 out-of-scope questions and 0.167 on the 6 false-premise ones. The low number is "
              "correct behaviour: on a false premise the system should correct the user instead of refusing. "
              "The combined 0.550 mixes two different expectations and should not be used as one "
              "number.")

    heading(doc, "6. Conclusions")
    bullet(doc, "Retrieval is the weak part of the system and generation is the strong one. Answer Relevancy is "
                "0.934 and 0.961, Faithfulness is 0.926 and 0.972, but Context Relevancy is 0.483 "
                "and 0.364. The model answers well on whatever it is given.")
    bullet(doc, "Retrieval returns a fixed five chunks for every question. On narrow factual "
                "questions four of them are usually irrelevant. Making the number of chunks depend "
                "on the question would be the first thing to try.")
    bullet(doc, "One real defect was found: the system agreed with a false premise about deleting a "
                "warehouse with products. The other five false-premise questions were handled "
                "correctly, so this looks like a single case rather than a habit.")
    bullet(doc, "Two frameworks on the same data differ by 0.026 on Answer Relevancy and 0.119 on "
                "Context Relevancy. The 0.444 gap on Faithfulness came from a bug inside EvaliQA. "
                "After the fix the gap is 0.046. One framework alone would have hidden this.")
    bullet(doc, "The judge model decides whether a metric works at all. The same metric is broken "
                "on gpt-4o-mini and fine on gpt-4.1-mini, and that is worth checking before "
                "trusting any number.")
    bullet(doc, "Fifty questions give a coarse picture: one case is worth 0.02. The numbers here "
                "point at places worth looking into. They do not measure the system in absolute terms.")

    doc.save(OUT)
    print("Saved:", OUT)

    words = sum(len(p.text.split()) for p in doc.paragraphs)
    print(f"Paragraphs: {len(doc.paragraphs)}, tables: {len(doc.tables)}, words: {words}")


if __name__ == "__main__":
    main()
