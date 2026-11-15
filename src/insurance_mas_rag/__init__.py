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
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver

# ---------- Settings ----------
PROJECT_DIR = Path(__file__).resolve().parents[2]   # C:\dev\insurance-mas-rag
PDF_DIR = PROJECT_DIR / "s3-rag-mas"
DB_DIR = PROJECT_DIR / "chroma_db"
COLLECTION = "mas_notices"
EMBED_MODEL = "text-embedding-3-large"   # changing this means re-ingesting
CHAT_MODEL = "gpt-4.1-mini"
K = 6                                    # passages retrieved per search

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
- Only include points that answer the question asked; a retrieved passage about a different topic is not relevant. Labels such as "Guideline 4.4" have their
  own numbering, separate from paragraphs.
- Never change "must", "should" or "may" when explaining: if the notice says "must", don't write "should", and
  never present a guideline as a "must". Keep a requirement and a nearby guideline in separate bullets.
- Cite paragraph numbers. Use the numbers printed in the passage text; the label shows which paragraph a passage
  starts in when it begins mid-paragraph.
- For numbers from tables (risk factors, percentages, rating classes), quote them exactly and name the table or
  appendix. If the table text looks garbled, say so instead of guessing.
- Explain jargon (e.g. C1, C2, HRG, CAR) in plain English, using the passages.
- Keep it concise. If the user asks what their firm should do, add one line saying this is information about the
  notices, not legal or compliance advice."""


def build_agent():
    """Create the chatbot agent. Conversation memory is kept per thread_id, in memory only."""
    vectorstore = open_vectorstore()

    @tool("search_notices", description=(
        "Search MAS Notice 125 and MAS Notice 133 for passages relevant to a query. "
        "Use specific keywords, e.g. 'Investment Committee composition' or 'premium liability risk factor'. "
        "Optionally set notice to 'MAS 125' or 'MAS 133' to search only that notice."
    ))
    def search_notices(query: str, notice: Literal["MAS 125", "MAS 133"] | None = None) -> str:
        where = {"notice": notice} if notice else None
        docs = vectorstore.similarity_search(query, k=K, filter=where)
        return format_chunks(docs) if docs else "No relevant passages found."

    return create_agent(
        model=init_chat_model(CHAT_MODEL, temperature=0.1),
        tools=[search_notices],
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
            q = c["args"].get("query", "")
            n = c["args"].get("notice")
            searches.append(f"{q} [{n}]" if n else q)
    passages = [m.content for m in turn if m.type == "tool"]
    return messages[-1].content, searches, passages
