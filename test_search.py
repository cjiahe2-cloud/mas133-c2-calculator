from docx import Document as WordDocument   # avoid clash with LangChain's Document
from docx.shared import Pt, RGBColor

from insurance_mas_rag import PROJECT_DIR, citation, load_env, open_vectorstore

load_env()

OUT_PATH = PROJECT_DIR / "search_results.docx"
K = 3                                            # results per query (fewer than the chatbot's, to keep the report short)

vectorstore = open_vectorstore()

print(f"Chunks in database: {vectorstore._collection.count()}\n")

# (query, where the answer should be) - the expected location is printed in the report so you can compare
queries = [
    # Direct questions
    ("Who must be on an insurer's Investment Committee?", "MAS 125 para 10"),
    ("What must an ALM policy include?", "MAS 125 para 21"),
    ("What are the supervisory solvency intervention levels?", "MAS 133 section 2"),
    ("What is the definition of adjusted capital ratio?", "MAS 133 para 1.5"),
    # Paraphrase - describes the concept without using its name
    ("Can an insurer use derivatives to speculate?", "MAS 125 derivatives paragraphs (around para 23-30)"),
    ("How should life policies be grouped when calculating insurance risk capital?", "MAS 133 paras 4.2.12-4.2.16"),
    # Table lookup
    ("Premium liability risk factor for high volatility category", "MAS 133 Appendix 4A Table 1 (136%)"),
    # Cross-reference between notices
    ("Meaning of derivative used in the investment notice", "MAS 125 para 6(d) -> MAS 133 definitions"),
    # Guidelines boxes - should be labelled 'Guideline', not 'para'
    ("What should an insurer consider for HRG stability?", "MAS 133 Guideline 4.4 (p.30)"),
    # Pages whose text had broken glyphs (missing N, Moody's) before page_text() fixed them
    ("Criteria for recognising an insurer's internal credit rating process", "MAS 133 Appendix 4E (p.140)"),
    ("Moody's rating to credit quality class mapping", "MAS 133 Appendix 4K (pp.162-163)"),
    # Appendix 3D, which numbers its paragraphs '(1)', '(2)'
    ("Can a callable bond be in the MA portfolio?", "MAS 133 Appendix 3D (around p.105)"),
    # Off-topic - should score badly (red)
    ("How do I bake sourdough bread?", "none"),
]


def score_colour(score: float) -> RGBColor:
    """Green = strong match, orange = related, red = weak. Thresholds are rough - tune after a first run."""
    if score < 0.85:
        return RGBColor(0x2E, 0x7D, 0x32)
    if score < 1.1:
        return RGBColor(0xEF, 0x6C, 0x00)
    return RGBColor(0xC6, 0x28, 0x28)


doc_out = WordDocument()
doc_out.add_heading("Retrieval Test Results", level=0)
doc_out.add_paragraph(
    f"Chunks in database: {vectorstore._collection.count()}  |  Top {K} results per query  |  "
    "Score = distance (lower is better): green < 0.85, orange < 1.1, red ≥ 1.1"
)

for q, expected in queries:
    results = vectorstore.similarity_search_with_score(q, k=K)

    # ---- Console: short summary ----
    print("=" * 80)
    print(f"QUERY: {q}\n  expected: {expected}")
    for doc, score in results:
        preview = " ".join(doc.page_content.split())[:120]
        print(f"  [{score:.3f}] {citation(doc.metadata)}: {preview}...")
    print()

    # ---- Word: full chunks ----
    doc_out.add_heading(f"Query: {q}", level=1)
    exp = doc_out.add_paragraph().add_run(f"Expected: {expected}")
    exp.italic = True
    for rank, (doc, score) in enumerate(results, start=1):
        header = doc_out.add_paragraph()
        run = header.add_run(f"#{rank}  score {score:.3f}  |  {citation(doc.metadata)}")
        run.bold = True
        run.font.color.rgb = score_colour(score)

        body = doc_out.add_paragraph(" ".join(doc.page_content.split()))
        body.runs[0].font.size = Pt(10)

    doc_out.add_page_break()

try:
    doc_out.save(OUT_PATH)
    print(f"Saved full results to {OUT_PATH}")
except PermissionError:
    print(f"Could not save - close {OUT_PATH.name} in Word and run again.")
