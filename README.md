# MAS Insurance Notices Chatbot

A question-answering chatbot for two Monetary Authority of Singapore notices:

| Notice | Title | Version in this project |
| --- | --- | --- |
| MAS 125 | Notice on Investments of Insurers | last revised 30 Sep 2022 |
| MAS 133 | Notice on Valuation and Capital Framework for Insurers | last revised 16 Mar 2026 |

Ask it a question and it searches the notices, then answers **only from the
passages it finds**, citing notice, page and paragraph (e.g.
`MAS 125 p.3, para 10`). If the notices don't cover something, it says so
instead of guessing.

It does **not** search the web. Its only source is the PDFs in `s3-rag-mas/`,
so it will not know about MAS revisions until you add them (see below).
Answers are information about the notices, not legal or compliance advice.

It comes in two forms: a chat in the terminal and a web app.

## How it works

1. **Ingest (once):** each PDF page is cleaned (running headers removed), split
   into ~1,200-character chunks, and tagged with the paragraph numbers and
   appendix it belongs to. The chunks are embedded with OpenAI's
   `text-embedding-3-large` and saved to a local [Chroma](https://www.trychroma.com/)
   database in `chroma_db/`.
2. **Ask:** an agent (`gpt-4.1-mini` via LangChain) decides what to search for
   (optionally just one notice), retrieves the 6 closest chunks, follows
   cross-references with further searches, and writes an answer with citations.
   It remembers the conversation, so follow-up questions work.

## Setup

You need [uv](https://docs.astral.sh/uv/) and an OpenAI API key.

1. Install dependencies:
   ```
   uv sync
   ```
2. Copy `.env.example` to `.env` and replace the placeholder with your key:
   ```
   OPENAI_API_KEY=sk-...
   ```
   `.env` is git-ignored. Never commit it.
3. Check the PDFs are in `s3-rag-mas/`. Filenames must be `<name>_<revision date>.pdf`,
   e.g. `MAS Notice 133_16 Mar 2026.pdf`: the number gives the notice name and
   the date gives the version.
4. Optional: preview the chunks and citations without calling OpenAI:
   ```
   uv run python ingest.py --dry-run
   ```
5. Build the database (this sends every chunk to OpenAI to be embedded, which
   costs a few cents for these two notices):
   ```
   uv run python ingest.py
   ```

### When MAS revises a notice

Replace the old PDF in `s3-rag-mas/` with the new one, then re-embed that notice:

| What | Command |
| --- | --- |
| Add a new notice | Put the PDF in `s3-rag-mas/`, then `uv run python ingest.py` |
| Replace a revised notice | `uv run python ingest.py --rebuild "MAS 133"` |
| Re-embed everything | `uv run python ingest.py --rebuild` |

If you add a revised PDF and forget `--rebuild`, `ingest.py` warns you that
an older revision is still in the store. Also update the versions in
`SYSTEM_PROMPT` (`src/insurance_mas_rag/__init__.py`) and in this README.

## Use

| What | Command |
| --- | --- |
| Web app (opens at http://localhost:8501) | `uv run streamlit run app.py` |
| Terminal chat (type `quit` to exit) | `uv run python chat.py` |

Conversations are kept in memory only and are lost when you close the app.

## Checking quality

Two scripts write a Word report for you to read through. Both call the OpenAI
API, so each run costs a little.

| Script | What it checks | Output |
| --- | --- | --- |
| `uv run python test_search.py` | Does search find the right passages? | `search_results.docx` |
| `uv run python answer_test.py` | Are the answers right, and are off-topic questions refused? | `answers.docx` |

The test questions include table lookups from MAS 133's appendices. PDF tables
don't always come out as clean text, so check those answers closely.

If a script prints "Could not save", close the `.docx` file in Word and run it again.

## Project layout

```
src/insurance_mas_rag/__init__.py   shared settings, prompt and agent (change models here)
ingest.py                           PDFs -> chroma_db/ (with paragraph tagging)
chat.py                             terminal chat
app.py                              web app (Streamlit)
test_search.py                      retrieval report
answer_test.py                      answer report
.env.example                        template for .env
.streamlit/config.toml              web app theme
```

If you change `EMBED_MODEL` in `src/insurance_mas_rag/__init__.py`, delete
`chroma_db/` and re-run `ingest.py`. Questions and chunks must be embedded by
the same model, and a different model can produce vectors of a different size,
which the existing database won't accept.
