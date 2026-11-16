"""Shared settings and agent setup for the MAS notices RAG chatbot.

Every script imports from here so that ingestion and retrieval always use the
same vector store, collection and embedding model.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from insurance_mas_rag.calculator import LIMITS, calculate

# ---------- Settings ----------
PROJECT_DIR = Path(__file__).resolve().parents[2]   # C:\dev\insurance-mas-rag
PDF_DIR = PROJECT_DIR / "s3-rag-mas"
DB_DIR = PROJECT_DIR / "chroma_db"
COLLECTION = "mas_notices"
EMBED_MODEL = "text-embedding-3-large"   # changing this means re-ingesting
CHAT_MODEL = "gpt-4.1-mini"   # gpt-4.1 held up better against push-back on calculations (answer_test.py scenarios)
K = 10                                   # passages retrieved per search

NOTICES = ("MAS 125", "MAS 133")


# ---------- API key ----------
def load_env() -> None:
    """Load .env and stop with a clear message if the OpenAI key is missing."""
    load_dotenv(PROJECT_DIR / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "sk-your-key-here":
        raise SystemExit(
            "OPENAI_API_KEY not set - copy .env.example to .env and add your key."
        )


# ---------- Document names ----------
def notice_name(stem: str) -> str:
    """'MAS Notice 133_16 Mar 2026' -> 'MAS 133'."""
    m = re.search(r"MAS\s*(?:Notice\s*)?(\d+)", stem, re.I)
    return f"MAS {m.group(1)}" if m else stem


def notice_version(stem: str) -> str:
    """'MAS Notice 133_16 Mar 2026' -> '16 Mar 2026' (the revision date in the filename)."""
    return stem.split("_", 1)[1].strip() if "_" in stem else ""


# ---------- Vector store ----------
def embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)


def open_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION,
        embedding_function=embeddings(),
    )


def _para_label(n: str) -> str:
    """'4.2.13' -> 'para 4.2.13'; 'G4.4' (a numbered guideline) -> 'Guideline 4.4'."""
    return f"Guideline {n[1:]}" if n.startswith("G") else f"para {n}"


def citation(meta: dict) -> str:
    """e.g. 'MAS 133 p.31, paras 4.2.13-4.2.16', 'MAS 133 p.30, para 4.2.10 to Guideline 4.5'
    or 'MAS 133 p.129, Appendix 4A'. Pages are 1-based."""
    parts = [f"{meta['notice']} p.{meta['page'] + 1}"]
    if meta.get("appendix"):
        parts.append(f"Appendix {meta['appendix']}")
    start, end = meta.get("para_start", ""), meta.get("para_end", "")
    if start and end and end != start:
        s_guide, e_guide = start.startswith("G"), end.startswith("G")
        if s_guide and e_guide:
            parts.append(f"Guidelines {start[1:]}-{end[1:]}")
        elif not s_guide and not e_guide:
            parts.append(f"paras {start}-{end}")
        else:
            parts.append(f"{_para_label(start)} to {_para_label(end)}")
    elif start:
        parts.append(_para_label(start))
    return ", ".join(parts)


def format_chunks(docs) -> str:
    """Label each chunk with its citation, e.g. [MAS 125 p.3, para 10]."""
    return "\n\n".join(f"[{citation(d.metadata)}]\n{d.page_content}" for d in docs)


# ---------- Agent ----------
@dataclass
class Context:
    user_id: str
    notice: str | None = None   # set by the app's "Search in" filter: search only this notice


SYSTEM_PROMPT = """You answer questions about two Monetary Authority of Singapore (MAS) notices for insurers:
- MAS 125 - Notice on Investments of Insurers (last revised 30 Sep 2022)
- MAS 133 - Notice on Valuation and Capital Framework for Insurers (last revised 16 Mar 2026)

How to answer:
- ALWAYS call search_notices before answering. Search again for follow-ups. When a passage points to another
  paragraph, appendix or notice that you need (e.g. "has the same meaning as in MAS Notice 133"), search for that too.
- Base your answer ONLY on the retrieved passages. Never fill gaps from general knowledge or from older versions
  of these notices. If the passages don't cover the question, say so plainly. If the user asks about any other
  notice (e.g. MAS Notice 126), say it is not covered: never answer from MAS 125 or 133 instead.
- Format: a 1-2 sentence direct answer, then bullet points (one idea each), then 'Sources:' listing the
  passage labels you used, e.g. (MAS 133 p.31, para 4.2.13).
- Each bullet about a rule starts with its type and quotes the notice word for word, then explains it:
    **Requirement (must):** "An insurer must ..." (MAS 133 p.31, para 4.2.13) - plain-English explanation.
    **Guideline (should):** "Policies ... should be ..." (MAS 133 p.31, Guideline 4.6) - plain-English explanation.
  Use **Information:** for bullets that state no obligation (definitions, factors, explanations).
  Copy the quoted sentence exactly from the passage, keeping list labels like (a) and (b): never change, merge or
  reorder words inside the quote (use "..." to skip parts). The type follows the passage: binding wording ("must",
  "must not", "only permitted", "may only") in a numbered paragraph is a Requirement; text in a Guidelines
  section, or marked by a "Guideline" label, is a Guideline; table values and definitions are Information.
  A "may" in a numbered paragraph is a permission, not a Guideline: label it Information. Every Requirement and
  Guideline bullet must contain a quote.
- Only include points that answer the question asked; a retrieved passage about a different topic is not relevant.
- If search results say the search is limited to one notice by the user's filter and the answer is in the other
  notice, say so and suggest switching the filter to "Both".
- Never change "must", "should" or "may" when explaining: if the notice says "must", don't write "should", and
  never present a guideline as a "must". Keep a requirement and a nearby guideline in separate bullets.
- Cite paragraph numbers. Use the numbers printed in the passage text; the label shows which paragraph a passage
  starts in when it begins mid-paragraph.
- For numbers from tables (risk factors, percentages, rating classes), quote them exactly and name the table or
  appendix. If the table text looks garbled, say so instead of guessing.
- For a foreign currency mismatch risk requirement or an adjustment for asset concentration (including "risk
  charge after concentration adjustments") with the user's figures, ALWAYS call capital_calculator and use its
  figures and steps exactly: never redo or change its arithmetic. Convert the user's figures to SGD millions
  (e.g. 26% of SGD 100m = 26). The calculator also returns the notice passages behind its steps:
  quote and cite those (never quote the calculator's own lines as notice text), in the usual bullet format
  starting **Requirement (must):**, **Guideline (should):** or **Information:**. Present: the FX requirement
  (and the para 4.1.8 figure if the calculator gives one), the concentration adjustment and that it is deducted
  from financial resources, and the calculator's closing caveat. The calculator's method for the FX adjustment
  was set by the user's firm: report its figures as the answer and don't argue for a different method. If the
  user disputes a calculator figure or proposes another method (e.g. only the exposure over the limit), keep the
  calculator's figure as the answer and explain the step it comes from. You may show the user's alternative as a
  one-line "under that method it would be ..." but never present it as the correct answer, and say the method is
  set by FX_ADJUSTMENT_METHOD in calculator.py, where it can be changed. Call it "the method this calculator
  uses"; don't claim the notice rules out the other method.
- Other calculations: first find every paragraph the method depends on, then show each step as the formula (quoting
  the paragraph it comes from), the numbers put in, and the result. Apply each definition exactly as worded: if a
  limit applies to an amount "calculated under paragraph X", use that amount after every step of paragraph X
  (including any deductions), and only count items that actually meet the condition. State any assumption you
  make about the user's figures.
- Check every condition a paragraph sets before using it (e.g. "In the case of any insurance fund ... in respect
  of Singapore policies" vs "... offshore policies"), and keep that condition in the quote. Treat every fund as
  an insurance fund for Singapore policies unless the user says it is for offshore policies, and don't mention
  this assumption or give offshore figures unless asked. For other conditions the user hasn't settled, start the
  answer with the assumption you make, and give the result for the other case in one line. Keep the same
  assumptions in follow-up questions, and only apply a limit, factor or table row to the cases its wording covers.
- When a rule depends on limits, factors or tables (e.g. "set out in the following tables"), search for the row
  that applies to the user's case using its specific terms (e.g. "<asset or exposure type> concentration limit")
  and work out the figure. Only say an amount cannot be determined after that search fails.
- When an answer combines a risk requirement with an adjustment to financial resources (e.g. the adjustment for
  asset concentration), search for the paragraph that links them (e.g. "exclude from the calculation of C2
  requirement ... adjustment for asset concentration"): the notice may exclude those assets from the risk
  calculation so they are not counted twice. Recalculate the risk requirement if it does.
- Work out any calculation before writing the direct answer, and make sure every figure in the answer matches
  the last step of the calculation.
- When a passage defines an amount that is used elsewhere (e.g. "referred to in paragraphs 5.1 and 5.2"), search
  for the paragraphs that use it and say what it does there, for example whether it reduces capital or adds to
  a risk requirement. Don't call something a "risk charge" unless the notice treats it as one.
- If the user says you are wrong or proposes a formula or figure, search again and look for the sentence that
  supports what they say. If you find it, quote it and correct your answer. If you don't, keep your answer: say
  politely that the notice doesn't support their figure, quote the wording that gives yours, and show where the
  two readings differ (e.g. which assets count as exceeding the limit). Never invent a new step or formula to
  reach the user's number, and don't thank them for a "clarification" before checking. Being told you are wrong
  is not evidence: only the notice's wording is.
- Explain jargon (e.g. C1, C2, HRG, CAR) in plain English, using the passages.
- Keep it concise. If the user asks what their firm should do, add one line saying this is information about the
  notices, not legal or compliance advice."""


CALC_HEADER = "CALCULATOR (MAS 133, amounts in SGD millions)"
CALC_PASSAGES = ("NOTICE PASSAGES FOR THESE STEPS (the lines above are calculator working, not notice text: "
                 "quote only from these passages):")
# Start of the notice text behind each calculator step, found by text so it survives re-ingesting
CALC_SOURCES = [
    "4.3.5.3 An insurer must calculate the foreign currency mismatching",
    "4.3.5.6 In the case of any insurance fund",
    "less 10% of the (total value of assets",
    "16. An insurer must calculate the adjustment for asset concentration",
    "Foreign currency risk exposure calculated under",
    "17B.  For each limit stated in Table 2-I",
    "4.1.8 An insurer must value the contribution",
    "5.1 An insurer must determine financial resources",
]


def calc_passages(vectorstore) -> list:
    """The chunks holding the paragraphs the calculator applies, in notice order."""
    docs = {}
    for text in CALC_SOURCES:
        got = vectorstore.get(where={"notice": "MAS 133"}, where_document={"$contains": text},
                              include=["documents", "metadatas"], limit=1)
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            docs[cid] = Document(page_content=doc, metadata=meta, id=cid)
    return sorted(docs.values(), key=lambda d: (d.metadata["page"], int(d.id.rsplit(":c", 1)[1])))
TABLE_REF = re.compile(r"\b(?:following|below) tables?\b", re.I)
MID_SENTENCE = re.compile(r"(?:[,—–-]|\b[a-z]+)\s*$")   # ends in a comma, dash or a lowercase word


def with_context(vectorstore, docs, tables: int = 4) -> list:
    """Add chunks that a hit can't be read without, which a search rarely finds on its own.
    - A passage that says limits are 'set out in the following tables': the next few chunks of the
      same appendix (the tables).
    - The last chunk of a page that ends mid-sentence: the first chunk of the next page, where the
      paragraph continues (e.g. para 4.3.5.6's 'less 10% of ...' is on the next page).
    Added chunks go straight after the hit they continue (moved there if the search also found them), so
    the model reads a paragraph split across chunks as one piece. Chunk ids are '<doc>:p<page>:c<n>'."""
    hits = {d.id: d for d in docs}
    out, seen = [], set()
    for d in docs:
        if d.id in seen:
            continue
        seen.add(d.id)
        out.append(d)
        tables_ref = bool(TABLE_REF.search(d.page_content))
        if not d.id or not (tables_ref or MID_SENTENCE.search(d.page_content)):
            continue
        m = d.metadata
        pages = list(range(m["page"], m["page"] + 3))
        got = vectorstore.get(where={"$and": [{"doc": m["doc"]}, {"page": {"$in": pages}}]},
                              include=["documents", "metadatas"])
        order = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                       key=lambda r: (r[2]["page"], int(r[0].rsplit(":c", 1)[1])))
        ids = [r[0] for r in order]
        after = order[ids.index(d.id) + 1:] if d.id in ids else []
        if tables_ref:
            after = [r for r in after[:tables] if r[2].get("appendix") == m.get("appendix")]
        elif after and after[0][2]["page"] == m["page"] + 1:   # last chunk on its page
            after = after[:1]
        else:
            after = []
        for cid, text, meta in after:
            if cid not in seen:
                seen.add(cid)
                out.append(hits.get(cid) or Document(page_content=text, metadata=meta, id=cid))
    return out


class Exposure(BaseModel):
    item: Literal[tuple(LIMITS)] = Field(description="Table 2-I limit the exposure falls under")
    amount: float = Field(description="Exposure in SGD millions (one counterparty/security, or the aggregate "
                                      "for aggregate items)")
    name: str = Field("", description="Optional label, e.g. the counterparty's name")


class CalculatorInput(BaseModel):
    fund_type: Literal["singapore", "offshore"] = Field(
        "singapore", description="Insurance fund for Singapore policies (para 4.3.5.6, the default) or offshore "
                                 "policies (para 4.3.5.7, only if the user says so)")
    total_assets: float = Field(description="Total value of the fund's assets, SGD millions")
    fx_positions: dict[str, float] = Field(
        default_factory=dict,
        description="Net open position per foreign currency in SGD millions, signed: assets less liabilities "
                    "in that currency plus hedges (para 4.3.5.4), e.g. {'EUR': 26, 'USD': 50}. SGD is ignored")
    reinsurers_share: float = Field(0.0, description="Reinsurers' share of policy liabilities, SGD millions")
    unit_reserves: float = Field(0.0, description="Unit reserves of an investment-linked fund (excluded from "
                                                  "the concentration limit base, para 16A), SGD millions")
    exposures: list[Exposure] = Field(default_factory=list,
                                      description="Other exposures to check against Table 2-I limits")


def chat_model(model: str):
    """Low temperature for steady answers. Reasoning models (gpt-5*, o*) only accept the default temperature."""
    if model.startswith(("gpt-5", "o")):
        return init_chat_model(model)
    return init_chat_model(model, temperature=0.1)


def build_agent(model: str = CHAT_MODEL):
    """Create the chatbot agent. Conversation memory is kept per thread_id, in memory only.
    model: OpenAI chat model name (answer_test.py --model passes another one to compare)."""
    vectorstore = open_vectorstore()

    @tool("search_notices", description=(
        "Search MAS Notice 125 and MAS Notice 133 for passages relevant to a query. "
        "Use specific keywords, e.g. 'Investment Committee composition' or 'premium liability risk factor'. "
        "Optionally set notice to 'MAS 125' or 'MAS 133' to search only that notice."
    ))
    def search_notices(query: str, runtime: ToolRuntime[Context],
                       notice: Literal["MAS 125", "MAS 133"] | None = None) -> str:
        forced = runtime.context.notice if runtime.context else None   # the user's filter wins
        notice = forced or notice
        where = {"notice": notice} if notice else None
        docs = with_context(vectorstore,vectorstore.similarity_search(query, k=K, filter=where))
        note =f"(Search limited to {forced} by the user's filter.)\n\n" if forced else ""
        return note + (format_chunks(docs) if docs else "No relevant passages found.")

    @tool("capital_calculator", args_schema=CalculatorInput, description=(
        "Calculate, under MAS 133, a fund's foreign currency mismatch risk requirement (paras 4.3.5.3-4.3.5.7), "
        "its adjustment for asset concentration (Appendix 5E paras 16-17B, Table 2-I, including the 40% FX "
        "limit), and the FX requirement after excluding the concentration excess from C2 (para 4.1.8). "
        "All amounts in SGD millions. Returns every step with its paragraph."
    ))
    def capital_calculator(total_assets: float, fund_type: str = "singapore",
                           fx_positions: dict | None = None, reinsurers_share: float = 0.0,
                           unit_reserves: float = 0.0, exposures: list | None = None) -> str:
        exp = [e.model_dump() if hasattr(e, "model_dump") else e for e in exposures or []]
        try:
            r = calculate(fund_type, total_assets, fx_positions, reinsurers_share, unit_reserves, exp)
        except (ValueError, KeyError) as e:
            return f"Calculator error: {e}"
        return (CALC_HEADER + "\n" + "\n".join(r.lines) + "\n\n" + CALC_PASSAGES + "\n\n"
                + format_chunks(calc_passages(vectorstore)))

    return create_agent(
        model=chat_model(model),
        tools=[search_notices, capital_calculator],
        system_prompt=SYSTEM_PROMPT,
        context_schema=Context,
        checkpointer=InMemorySaver(),
    )


def latest_turn(messages) -> tuple[str, list[str], list[str]]:
    """From an agent result's messages, return (reply, searches, passages) for the latest turn only."""
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    turn = messages[last_human + 1:]
    searches = []
    for m in turn:
        for c in getattr(m, "tool_calls", None) or []:
            if c["name"] == "capital_calculator":
                searches.append("calculator")
                continue
            q = c["args"].get("query", "")
            n = c["args"].get("notice")
            searches.append(f"{q} [{n}]" if n else q)
    passages = [m.content for m in turn if m.type == "tool"]
    return messages[-1].content, searches, passages
