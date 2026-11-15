"""Answer check -> answers.docx (or answers_<model>.docx with --model).

    uv run python answer_test.py                      # one-shot questions + chatbot scenarios
    uv run python answer_test.py --model gpt-4.1      # same, with another model, to compare
    uv run python answer_test.py --scenarios-only     # only the multi-turn chatbot scenarios
"""
import argparse
import re
import uuid
from typing import Literal

from docx import Document as WordDocument
from docx.shared import Pt, RGBColor
from pydantic import BaseModel, Field

from insurance_mas_rag import (CHAT_MODEL, K, PROJECT_DIR, Context, build_agent, chat_model, format_chunks,
                               latest_turn, load_env, open_vectorstore)

parser = argparse.ArgumentParser(description="Check chatbot answers and write a Word report.")
parser.add_argument("--model", default=CHAT_MODEL, help=f"chat model to test (default {CHAT_MODEL})")
parser.add_argument("--scenarios-only", action="store_true", help="skip the one-shot questions")
args = parser.parse_args()

load_env()

MODEL = args.model
OUT_PATH = PROJECT_DIR / ("answers.docx" if MODEL == CHAT_MODEL else f"answers_{MODEL}.docx")
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


llm = chat_model(MODEL).with_structured_output(Answer)


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
  other text with no obligation. A "may" in a numbered paragraph is a permission, not a guideline: "information".
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

# ---- Multi-turn scenarios, run through the real chatbot (search tool, follow-up searches, memory) ----
# Each turn: (user message, [(what is checked, regex, should it match?)]). The regexes look for the key facts
# a correct answer must state; read the full replies in the report too.
M = r"\s*(?:m\b|mn\b|million)"
SCENARIOS = {
    "FX mismatch and asset concentration (MAS 133 para 4.3.5.6, Appendix 5E, para 5.1)": [
        ("My insurance fund for Singapore policies has total assets of SGD 100m and zero liabilities, invested "
         "20% SGD, 30% EUR and 50% USD, with no hedges. Calculate the foreign currency mismatch risk requirement "
         "and give it as a % of market value.",
         [("FX risk exposure of 70m (80m less 10% of assets)", r"\b70" + M, True),
          ("risk requirement of 8.4m / 8.4%", r"\b8\.40?(?!\d)", True)]),
        # The FX adjustment follows calculator.FX_ADJUSTMENT_METHOD ("total_less_limit"): 70m exposure is over the
        # 40m limit, so 100m - 40m = 60m. With method "excess" it would be 70m - 40m = 30m.
        ("Is there an adjustment for asset concentration for this fund? How much is it?",
         [("adjustment of 60m (100m total assets less the 40m limit)", r"\b60" + M, True)]),
        ("I think the adjustment is an additional risk charge on top of the 8.4m, right?",
         [("says it is deducted from financial resources (para 5.1)", r"financial resources", True),
          ("does not simply agree", r"^\W*(yes|correct|that's right|you are (right|correct)|you're (right|correct))\b", False)]),
    ],
    # Fund type not stated: treated as Singapore policies (para 4.3.5.6, less 10%) without saying so (user's choice).
    # Exposure 76m - 10m = 66m, requirement 7.92m; 66m is over the 40m limit, so adjustment 100m - 40m = 60m.
    "FX with unstated fund type, then push-back to the excess method": [
        ("I have a SGD100m portfolio with 24% SGD, 26% EUR, 50% USD. What is my risk charge % MV?",
         [("doesn't bring up offshore policies", r"offshore", False),
          ("gives 7.92% for a Singapore-policies fund", r"\b7\.92\b", True)]),
        ("I only want the concentration adjustment for FX risk, based on the portfolio I mentioned earlier.",
         [("adjustment of 60m (100m less the 40m limit)", r"\b60" + M, True)]),
        ("This is not correct. It should only be the exposure over the limit: 66m - 40m = 26m.",
         [("keeps 60m", r"\b60" + M, True),
          ("does not cave", r"^\W*(yes|correct|you are (right|correct)|you're (right|correct)|thank you for "
                            r"(your|the) clarification)\b", False),
          ("does not conclude 26m", r"(correct|the) adjustment (for [^.\n]{0,30})?(is|=)\s*(SGD\s*)?26" + M, False)]),
    ],
    "FX risk charge after the concentration adjustment": [
        ("I have a SGD100m portfolio with 24% SGD, 26% EUR, 50% USD, in an insurance fund for Singapore policies. "
         "What is my risk charge % MV, after FX risk concentration adjustments?",
         [("FX requirement 7.92% (no para 4.1.8 recalculation under this method)", r"\b7\.92\b", True),
          ("concentration adjustment of 60m", r"\b60" + M, True),
          ("deducted from financial resources", r"financial resources", True)]),
    ],
}

doc_out = WordDocument()
doc_out.add_heading(f"Chatbot Answers ({MODEL})", level=0)
doc_out.add_paragraph("Each point shows its type, the quote from the notice, and an automatic check: "
                      "✓ the quote appears word for word in the retrieved passages and its type matches; "
                      "✗ something to look at.")
total = flagged = 0

for q in ([] if args.scenarios_only else questions):
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

if not args.scenarios_only:
    print(f"Checks: {total - flagged} of {total} points passed, {flagged} flagged.\n")
    doc_out.add_paragraph(f"Checks: {total - flagged} of {total} points passed, {flagged} flagged.")

agent = build_agent(MODEL)
s_total = s_failed = 0
doc_out.add_heading("Chatbot scenarios (multi-turn)", level=0)
for name, turns in SCENARIOS.items():
    print(f"Scenario: {name}")
    doc_out.add_heading(name, level=1)
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    for user_msg, expectations in turns:
        result = agent.invoke({"messages": [{"role": "user", "content": user_msg}]},
                              config=thread, context=Context(user_id="test"))
        reply, searches, _ = latest_turn(result["messages"])
        print(f"  USER: {user_msg}\n  SEARCHES: {' | '.join(searches)}\n  REPLY: {reply}\n")

        run = doc_out.add_paragraph().add_run(f"User: {user_msg}")
        run.bold = True
        run = doc_out.add_paragraph().add_run(f"Searches: {' | '.join(searches)}")
        run.font.size = Pt(9)
        run.font.color.rgb = GREY
        doc_out.add_paragraph(reply)
        for what, pattern, should_match in expectations:
            ok = bool(re.search(pattern, reply, re.I | re.M)) == should_match
            s_total += 1
            s_failed += not ok
            print(f"    {'✓' if ok else '✗'} {what}")
            run = doc_out.add_paragraph(style="List Bullet").add_run(f"{'✓' if ok else '✗'} {what}")
            run.bold = True
            run.font.color.rgb = GREEN if ok else RED
        print()

print(f"Scenario checks: {s_total - s_failed} of {s_total} passed.")
doc_out.add_paragraph(f"Scenario checks: {s_total - s_failed} of {s_total} passed.")
try:
    doc_out.save(OUT_PATH)
    print(f"Saved to {OUT_PATH}")
except PermissionError:
    print(f"Could not save - close {OUT_PATH.name} in Word and run again.")
