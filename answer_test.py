import re
from typing import Literal

from docx import Document as WordDocument
from docx.shared import Pt, RGBColor
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from insurance_mas_rag import CHAT_MODEL, K, PROJECT_DIR, format_chunks, load_env, open_vectorstore

load_env()

OUT_PATH = PROJECT_DIR / "answers.docx"
RED, GREEN, GREY = RGBColor(0xC6, 0x28, 0x28), RGBColor(0x2E, 0x7D, 0x32), RGBColor(0x75, 0x75, 0x75)

vectorstore = open_vectorstore()


# ---- The shape of the answer we want back ----
class Point(BaseModel):
    type: Literal["requirement", "guideline", "information"] = Field(
        description="requirement = binding wording ('must', 'must not', 'only permitted', 'may only') in a "
                    "numbered paragraph; guideline = text in a Guidelines section or under a 'Guideline' label; "
                    "information = states no obligation (definitions, table values, explanations)")
    quote: str = Field(description="The supporting sentence copied word for word from one excerpt, keeping "
                                   "list labels like (a) and (b). Use '...' to skip parts, but never change, "
                                   "merge or reorder words inside the quote")
    source: str = Field(description="Label of the excerpt the quote is from, e.g. 'MAS 133 p.31, para 4.2.13'")
    explanation: str = Field(description="Plain-English explanation. Keep the notice's must/should/may")


class Answer(BaseModel):
    summary: str = Field(description="One or two plain-English sentences that directly answer the question")
    key_points: list[Point] = Field(description="2-6 points, one idea each")
    sources: list[str] = Field(description="Passage labels used, formatted like 'MAS 125 p.3, para 10'")
    covered: bool = Field(description="False if the excerpts don't actually answer the question")


llm = init_chat_model(CHAT_MODEL, temperature=0.1).with_structured_output(Answer)


def answer(question: str) -> tuple[Answer, list]:
    docs = vectorstore.similarity_search(question, k=K)
    prompt = f"""You answer questions about MAS Notice 125 and MAS Notice 133 using ONLY the excerpts below.

Rules:
- Plain English; explain jargon.
- For every point, copy the supporting sentence word for word from one excerpt into 'quote', keeping list labels
  like (a) and (b). Use '...' to skip parts; never change, merge or reorder words inside the quote. Give that
  excerpt's label exactly as shown in [brackets] as 'source'.
- type: "requirement" for binding wording ("must", "must not", "only permitted", "may only") in a numbered
  paragraph; "guideline" for text in a Guidelines section or an excerpt labelled "Guideline" (labels such as
  "Guideline 4.4" are numbered separately from paragraphs); "information" for definitions, table values and
  other text with no obligation.
- Only include points that answer the question asked. An excerpt about a different policy or topic is not
  relevant just because it was retrieved.
- Never change "must", "should" or "may" in the explanation. Keep a requirement and a nearby guideline as
  separate points.
- Quote numbers from tables exactly and name the table or appendix.
- If the excerpts don't answer the question, set covered=false, give no points, and say so in the summary.
  Never use outside knowledge.
- The excerpts come only from MAS 125 and MAS 133. If the question is about any other notice (e.g. MAS Notice 126),
  set covered=false: never answer it from a different notice, even one with a similar number.

QUESTION: {question}

EXCERPTS:
{format_chunks(docs)}"""
    return llm.invoke(prompt), docs


# ---- Automatic checks on each point ----
BINDING = re.compile(r"\b(must|shall|only permitted|may only|not permitted|is required|are required)\b")


def normalise(text: str) -> str:
    """Ignore differences that are not wording: line breaks, curly quotes, dashes, 'sub- paragraph',
    and footnote numbers stuck to a word ('asset-liability management2')."""
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    text = re.sub(r"(?<=[a-z])\d+\b", "", text)       # lowercase only, so 'C1' and 'AT1' are kept
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"-\s+", "-", text).strip().lower()


def check(point: Point, docs) -> list[str]:
    """Problems with a point: a quote not found word for word in the excerpts, or a type that doesn't match it."""
    problems = []
    passages = normalise(" ".join(d.page_content for d in docs))
    # '...' skips text; so does a line break ('20. ... its ALM policy:' then '(b) is appropriate...')
    parts = [normalise(p).strip(" .;,:\"'") for p in re.split(r"\.\.\.|…|\n", point.quote)]
    if not all(p in passages for p in parts if p):
        problems.append("quote not found word for word in the excerpts")
    binding = BINDING.search(point.quote)
    if point.type == "requirement" and not binding:
        problems.append("labelled requirement but the quote has no binding wording ('must', 'only permitted'...)")
    if point.type == "information" and binding:
        problems.append(f"quote says '{binding.group()}' but is labelled information")
    if point.type == "guideline" and "guideline" not in point.source.lower():
        problems.append("labelled guideline but the source is not a Guideline label (check)")
    return problems


questions = [
    "Who must be on an insurer's Investment Committee?",
    "What must an ALM policy include?",
    "What are the supervisory solvency intervention levels?",
    "Can an insurer use derivatives to speculate?",
    "What is the premium liability risk factor for the high volatility category?",
    "What is the adjusted capital ratio?",       # definition - should cite MAS 133 para 1.5
    "How should an insurer group policies into HRGs?",   # mixes 'must' paragraphs and 'should' guidelines
    "What does MAS Notice 126 require?",          # not in the corpus - should be covered=false
    "How do I bake sourdough bread?",             # off-topic - should be covered=false
]

doc_out = WordDocument()
doc_out.add_heading("Chatbot Answers", level=0)
doc_out.add_paragraph("Each point shows its type, the quote from the notice, and an automatic check: "
                      "✓ the quote appears word for word in the retrieved passages and its type matches; "
                      "✗ something to look at.")
total = flagged = 0

for q in questions:
    print(f"Answering: {q}")
    a, docs = answer(q)
    print(f"  covered={a.covered}  sources: {'; '.join(a.sources)}\n  {a.summary}")

    doc_out.add_heading(q, level=1)
    if not a.covered:
        run = doc_out.add_paragraph().add_run("Not covered in the notices.")
        run.bold = True
        run.font.color.rgb = RED
    doc_out.add_paragraph(a.summary)

    for point in a.key_points:
        problems = check(point, docs)
        total += 1
        flagged += bool(problems)
        mark = "✗ " + "; ".join(problems) if problems else "✓"
        print(f"    - [{point.type}] \"{point.quote}\" ({point.source})\n      {point.explanation}\n      {mark}")

        p = doc_out.add_paragraph(style="List Bullet")
        p.add_run(f"{point.type.capitalize()}: ").bold = True
        p.add_run(f"“{point.quote}” ").italic = True
        p.add_run(f"({point.source}) ").font.size = Pt(9)
        p.add_run(f"– {point.explanation} ")
        run = p.add_run(mark)
        run.bold = True
        run.font.color.rgb = RED if problems else GREEN
    print()

    run = doc_out.add_paragraph().add_run("Sources: " + "; ".join(a.sources))
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = GREY

print(f"Checks: {total - flagged} of {total} points passed, {flagged} flagged.")
doc_out.add_paragraph(f"Checks: {total - flagged} of {total} points passed, {flagged} flagged.")
try:
    doc_out.save(OUT_PATH)
    print(f"Saved to {OUT_PATH}")
except PermissionError:
    print(f"Could not save - close {OUT_PATH.name} in Word and run again.")
