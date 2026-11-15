"""Add the MAS notice PDFs in s3-rag-mas/ to the vector store.

Safe to re-run: notices already in the store are skipped, so only new PDFs are
embedded (and paid for).

    uv run python ingest.py                     # add any new PDFs
    uv run python ingest.py --rebuild "MAS 133" # re-embed one notice (e.g. after MAS revises it)
    uv run python ingest.py --rebuild           # re-embed every notice
    uv run python ingest.py --dry-run           # show chunks and citations without calling OpenAI

When MAS publishes a revision, replace the old PDF in s3-rag-mas/ (keep the
'<name>_<date>.pdf' naming) and run with --rebuild for that notice.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from insurance_mas_rag import PDF_DIR, citation, load_env, notice_name, notice_version, open_vectorstore

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, add_start_index=True)

# ---------- Page cleaning ----------
HEADER = re.compile(r"^(MAS Notice \d+\s*\S\s*Notice on .*|Notice on [A-Z][a-z].*|\d{1,3})$")
LONE_LABEL = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4}[A-Z]?\.?|\(?[a-z]{1,4}\)|[A-Z]\.)$")   # '17.', '(a)', 'b)', 'B.'
APPENDIX = re.compile(r"^Appendix\s+([0-9]*[A-Z]{0,2}(?:-[IVX]+)?)$")
SECTION = re.compile(r"^[A-Z]\.\s+[A-Z][A-Za-z ,’()'&/-]{3,90}$")   # 'F. Adjustment for Asset Concentration'


# Some Calibri glyphs in MAS 133 have a broken character map and come out as spaces:
# 'RECOG ITIO  OF I TER AL' (pp.136-155), 'op ons', 'e ec ve', 'equity like' (p.127), 'M  dy’s' (p.162).
# Keyed by (font, glyph id); the ids are Calibri's own, so they hold across pages.
GLYPH_FIX = {
    ("Calibri", 299): "ff", ("Calibri", 302): "fi", ("Calibri", 327): "fl", ("Calibri", 415): "ti",
    ("Calibri", 460): "z", ("Calibri", 845): "?", ("Calibri", 882): "-", ("Calibri", 891): "•",
    ("ArialMT", 135): "•",
    ("Calibri-Bold", 69): "N", ("Calibri-Bold", 286): "e", ("Calibri-Bold", 381): "o",
    ("Calibri-Bold", 400): "s", ("Calibri-Bold", 410): "t", ("Calibri-Bold", 855): ":",
}


def page_text(page, spaces: dict | None = None) -> str:
    """page.get_text(), with the broken glyphs in GLYPH_FIX restored. Rebuilt from the
    character-level output, which gives the same text as get_text() character for character.

    spaces, if given, collects {(font, glyph id): [(1-based page, width in em), ...]} for every
    glyph that comes out as a space and is not in GLYPH_FIX, for check_spaces()."""
    trace = page.get_texttrace()
    if spaces is not None:
        for span in trace:
            for u, gid, _, bbox in span["chars"]:
                if u == 32 and (span["font"], gid) not in GLYPH_FIX:
                    width = (bbox[2] - bbox[0]) / span["size"] if span["size"] else 0
                    spaces.setdefault((span["font"], gid), []).append((page.number + 1, width))
    fixes = {(round(o[0], 2), round(o[1], 2)): GLYPH_FIX[(span["font"], gid)]
             for span in trace
             for u, gid, o, _ in span["chars"]
             if u == 32 and (span["font"], gid) in GLYPH_FIX}
    if not fixes:
        return page.get_text()
    out = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            prev_x1 = None
            for span in line["spans"]:
                for c in span["chars"]:
                    ch = c["c"]
                    if ch == " ":
                        ch = fixes.get((round(c["origin"][0], 2), round(c["origin"][1], 2)), " ")
                    # Text placed by position (p.127's flowchart) has no space characters between
                    # words, so a visible gap is the word break
                    if (prev_x1 is not None and ch != " " and out and out[-1] != " "
                            and c["bbox"][0] - prev_x1 > 0.1 * span["size"]):
                        out.append(" ")
                    out.append(ch)
                    prev_x1 = c["bbox"][2]
            out.append("\n")
    return "".join(out)


def check_spaces(name: str, spaces: dict) -> None:
    """Warn about glyphs that come out as a space but are not the font's real space.

    In each font the real space is by far the most used space glyph. Another glyph that extracts
    as a space but is a different width is probably a letter with a broken character map (like
    the N in MAS 133, 0.66 em wide against the space's 0.23), so the stored text has holes in it.
    A second glyph of the same width is just another space (MAS 125's Times New Roman has two).
    To fix: find what the glyph is from the words around it and add it to GLYPH_FIX."""
    median = lambda xs: sorted(xs)[len(xs) // 2]
    by_font: dict[str, list[tuple[int, int, list]]] = {}
    for (font, gid), uses in spaces.items():
        by_font.setdefault(font, []).append((len(uses), gid, uses))
    for font, glyphs in by_font.items():
        glyphs.sort(reverse=True)
        space_width = median([w for _, w in glyphs[0][2]])     # glyphs[0] is the real space
        for count, gid, uses in glyphs[1:]:
            if abs(median([w for _, w in uses]) - space_width) <= 0.02:
                continue
            pages = sorted({p for p, _ in uses})
            shown = ", ".join(map(str, pages[:8])) + ("..." if len(pages) > 8 else "")
            print(f"  WARNING: {name}: glyph {gid} in font {font} extracts as a space "
                  f"{count} time(s), pages {shown}. It may be a letter with a broken font map: "
                  f"check those pages and add it to GLYPH_FIX.")


def clean_page(text: str) -> str:
    """Drop the running header and page number, join lone labels like '17.' or '(a)' to the
    line after them, and squeeze blank lines."""
    lines = [l.strip() for l in text.split("\n")]

    # Running header / page number sit in the first few non-empty lines
    seen = 0
    for i, l in enumerate(lines):
        if not l:
            continue
        seen += 1
        if seen > 4:
            break
        if HEADER.match(l):
            lines[i] = ""

    out: list[str] = []
    pending = ""
    for l in lines:
        if not l:
            if out and out[-1] and not pending:
                out.append("")
            continue
        if pending:
            l = f"{pending} {l}"
            pending = ""
        elif LONE_LABEL.match(l):
            pending = l
            continue
        out.append(l)
    if pending:
        out.append(pending)
    return "\n".join(out).strip()


# ---------- Paragraph numbers ----------
MARKER = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4}[A-Z]?)(\.?)\s+(\S.*)$")
GUIDE_HEAD = re.compile(r"^Guidelines\s*:?$")
MUST = re.compile(r"\bmust\b")
SUB_PARA = re.compile(r"^\(\d+\)")               # '(2) An insurer may...' in appendix tables
SUB_ITEM = re.compile(r"^\(?(?:[a-z]{1,2}|[ivx]{1,5})\)\s")   # 'd) An insurer must', '(ii) The insurer must'
# A definition line: optional '(d) ', a quoted term, then 'means' / 'has the same meaning' / ...
DEF_START = re.compile(r"^(?:\([a-z]{1,4}\)\s*)?“[^”]{1,100}”.{0,80}?\b(?:means|has|have|refers|includes)\b")


def first_sentence(text: str, following: list[str]) -> str:
    """text plus the lines after it, up to the first full stop. Stops early at a blank line or
    the next numbered paragraph, since a table cell or paragraph often follows without a gap."""
    for line in following:
        if re.search(r"\.(\s|$)", text) or len(text) > 400:
            break
        if not line or MARKER.match(line) or GUIDE_HEAD.match(line) or SUB_PARA.match(line) or SUB_ITEM.match(line):
            break
        text += " " + line
    return re.split(r"\.(?:\s|$)", text, maxsplit=1)[0]


def top_of(num: str) -> int:
    return int(re.match(r"\d+", num).group())


def guide_key(num: str) -> tuple[int, str]:
    """'4.5' -> (5, ''), '3.2A' -> (2, 'A'), '3' -> (3, ''): the guideline's place in its sequence."""
    m = re.fullmatch(r"(?:\d+\.)?(\d+)([A-Z]?)", num)
    return (int(m.group(1)), m.group(2))


class ParaTracker:
    """Finds paragraph numbers at the start of lines ('17. An insurer...', '4.2.13 An insurer...').

    Filters out look-alikes: table values ('0.25', '1.5 %'), wrapped cross-references
    ('4.2.18 to 4.2.28') and numbers from an unrelated chapter.

    MAS 133 also has numbered 'Guidelines:' boxes with their own sequence, mixed in with the
    paragraphs (4.2.10, Guidelines 4.4 and 4.5, 4.2.11...; in appendices 2., Guideline 1., 3.).
    These are returned with a 'G' prefix ('G4.4') so they are never cited as paragraphs."""

    def __init__(self):
        self.appendix = ""
        self.last_top: int | None = None   # chapter (MAS 133) or paragraph (MAS 125) of the last paragraph
        self.main = ""                     # last paragraph number (not guideline)
        self.current = ""                  # paragraph or 'G' guideline in effect
        self.in_guide = False              # inside a 'Guidelines:' box
        self.guide_last: tuple[int, str] | None = None   # last guideline in this chapter/appendix
        self.paren_paras = False           # this appendix numbers paragraphs '(1)', '(2)'

    def set_appendix(self, name: str) -> None:
        if name != self.appendix:
            self.appendix, self.last_top, self.main, self.current = name, None, "", ""
            self.in_guide, self.guide_last, self.paren_paras = False, None, False

    def scan(self, text: str) -> list[tuple[int, str]]:
        """Return (offset, paragraph number) for each paragraph or guideline that starts in text."""
        found, offset = [], 0
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if GUIDE_HEAD.match(line):
                self.in_guide = True
            elif self.appendix and SUB_PARA.match(line):
                # Appendix 3D numbers its paragraphs '(1)', '(2)'..., which are not tracked: after
                # a guideline, cite the appendix only rather than a wrong paragraph
                self.paren_paras = True
                if self.current.startswith("G"):
                    self.in_guide, self.current = False, ""
                    found.append((offset, ""))
            elif self.current.startswith("G") and (
                    DEF_START.match(line)
                    or (SUB_ITEM.match(line) and MUST.search(first_sentence(line, lines[i + 1:])))):
                # The Guidelines box has ended and the paragraph around it continues, with no
                # new number: the next definition, or a sub-item with 'must' (a requirement)
                self.in_guide = False
                self.current = "" if self.paren_paras else self.main
                found.append((offset, self.current))
            elif m := MARKER.match(line):
                num = m.group(1)
                is_guide = self.in_guide and self._guideline(m)
                if is_guide and self._accept(m) and self._looks_main(m, lines[i + 1:]):
                    is_guide = False             # fits both sequences: decide on the wording
                if is_guide:
                    self.guide_last = guide_key(num)
                    self.current = "G" + num
                    found.append((offset, self.current))
                elif self._accept(m):
                    top = top_of(num)
                    if "." in num and top != self.last_top:
                        self.guide_last = None   # each chapter numbers its guidelines from 1
                    self.in_guide = False
                    self.main = self.current = num
                    self.last_top = top
                    found.append((offset, num))
            offset += len(line) + 1
        return found

    def _guideline(self, m: re.Match) -> bool:
        """Is this the next number in the guideline sequence ('4.4' after '4.3', '3.2A' after '3.2')?"""
        num, dot, rest = m.groups()
        if not (rest[0].isupper() or rest[0] in "[\"“("):
            return False
        if "." in num:                           # chapter style: '4.5' inside chapter 4
            if num.count(".") != 1 or top_of(num) != self.last_top:
                return False
        elif not dot or not self.main or "." in self.main:
            return False                         # appendix style '3.' only where paragraphs are '3.'
        key = guide_key(num)
        if self.guide_last is None:
            return key == (1, "")
        return key > self.guide_last and key[0] - self.guide_last[0] in (0, 1)

    @staticmethod
    def _looks_main(m: re.Match, following: list[str]) -> bool:
        """For a number that could be either. In chapters ('4.5'), only a section heading
        ('4.3 C2 Requirement') is a paragraph: chapter paragraphs are numbered '4.2.13'. In
        appendices ('3.'), a first sentence with 'must' is a paragraph; guidelines say 'should'."""
        rest = m.group(3).strip()
        if len(rest) < 50 and rest[0] != "[" and not rest.endswith((".", ";", ":", ",", "]")):
            return True                          # short title line, not '[This paragraph has been deleted.]'
        if "." in m.group(1):
            return False
        return bool(MUST.search(first_sentence(rest, following)))

    def _accept(self, m: re.Match) -> bool:
        num, dot, rest = m.groups()
        if "." not in num and not dot:          # a bare '5' is a page number or table value
            return False
        top = top_of(num)
        if any(top_of(p) == 0 for p in num.split(".")):
            return False                         # '0.25', '5.0' etc. in tables
        if "." not in num and "." in self.main:
            return False                         # '7.' list item inside a document numbered '4.2.13'
        if not (rest[0].isupper() or rest[0] in "[\"“("):
            return False                         # '4.2.18 to 4.2.28', '1.5 %'
        if self.last_top is None:
            return top <= 2                      # a document or appendix starts at 1 (or 2)
        return top in (self.last_top, self.last_top + 1)


# ---------- Definitions ----------
def segments(text: str) -> list[tuple[int, str]]:
    """Split a page into (offset, text) pieces. On a page of definitions, each definition is its
    own piece, so 'adjusted capital ratio' gets a chunk of its own instead of sharing one with
    ten other terms. Other pages are one piece."""
    starts, offset = [], 0
    for line in text.split("\n"):
        if DEF_START.match(line):
            starts.append(offset)
        offset += len(line) + 1
    if len(starts) < 3:                          # a stray quoted term, not a list of definitions
        return [(0, text)]
    bounds = [0, *starts, len(text)] if starts[0] else [*starts, len(text)]
    return [(a, text[a:b]) for a, b in zip(bounds, bounds[1:]) if text[a:b].strip()]


# ---------- Loading ----------
def load_pdf(path: Path) -> list[Document]:
    """Chunks with notice, version, 0-based page, appendix and paragraph range in metadata."""
    doc, notice, version = path.stem, notice_name(path.stem), notice_version(path.stem)
    tracker = ParaTracker()
    chunks: list[Document] = []
    spaces: dict = {}                             # for check_spaces()
    app_title = section = ""                      # for appendix chunks' context line

    with pymupdf.open(path) as pdf:
        for page_no, page in enumerate(pdf):
            text = clean_page(page_text(page, spaces))
            if not text:
                continue

            lines = text.split("\n")
            for i, line in enumerate(lines[:6]):  # an appendix title sits at the top of its first page
                if m := APPENDIX.match(line):
                    tracker.set_appendix(m.group(1))
                    title = []
                    for t in lines[i + 1:i + 4]:  # the capitalised title lines after 'Appendix 5E'
                        if not (re.search(r"[A-Z]", t) and t == t.upper()):
                            break
                        title.append(t.strip())
                    app_title, section = " ".join(title)[:90], ""
                    break
            # Lettered sections inside an appendix ('F. Adjustment for Asset Concentration'), by offset
            sections, offset = [], 0
            for i, line in enumerate(lines):
                following = lines[i + 1] if i + 1 < len(lines) else ""
                # A heading is short and stands alone; a list item ('B. The simplified balance sheet
                # of the MA portfolio containing') wraps onto a line starting in lowercase
                if (tracker.appendix and SECTION.match(line) and len(line.strip()) <= 60
                        and not following[:1].islower()):
                    sections.append((offset, line.strip()))
                offset += len(line) + 1

            carried = tracker.current             # paragraph continuing from the previous page
            markers = tracker.scan(text)
            meta = {"doc": doc, "notice": notice, "version": version, "page": page_no,
                    "appendix": tracker.appendix, "source": path.name}

            pieces = [(seg_start + d.metadata["start_index"], d.page_content)
                      for seg_start, seg in segments(text)
                      for d in splitter.create_documents([seg])]
            for start, content in pieces:
                end = start + len(content)
                # Context line for appendix chunks: a table chunk ('Item Description of limit...') then
                # carries its appendix and section names, so a search for the section topic finds it
                section = next((s for off, s in reversed(sections) if off <= start + 60),   # starts in it
                               next((s for off, s in sections if off < end), section))   # or first inside
                if tracker.appendix:
                    context = f"Appendix {tracker.appendix}" + (f": {app_title}" if app_title else "")
                    content = f"[{context}{f' · {section}' if section else ''}]\n{content}"
                chunk = Document(page_content=content, metadata=dict(meta))
                before = [n for off, n in markers if off <= start]
                inside = [(off, n) for off, n in markers if start < off < end]
                para_start = before[-1] if before else carried
                # Only a heading ('Guidelines:', 'Homogeneous Risk Grouping') before the first
                # number: the chunk starts with that paragraph, not the one carried over
                # ('' from a marker means 'no number known', e.g. Appendix 3D's '(5)': keep it)
                if inside and (inside[0][0] - start <= 60 or (not before and not carried)):
                    para_start = inside[0][1]
                inside = [n for _, n in inside]
                chunk.metadata["para_start"] = para_start
                chunk.metadata["para_end"] = inside[-1] if inside else para_start
                chunks.append(chunk)
    check_spaces(notice, spaces)
    return chunks


def chunk_ids(chunks: list[Document]) -> list[str]:
    """Fixed IDs like 'MAS Notice 133_16 Mar 2026:p34:c2', so storing a chunk again replaces it."""
    seen = defaultdict(int)
    ids = []
    for c in chunks:
        key = (c.metadata["doc"], c.metadata["page"])
        ids.append(f"{key[0]}:p{key[1]}:c{seen[key]}")
        seen[key] += 1
    return ids


def matches(pdf: Path, name: str) -> bool:
    """'MAS 133', 'MAS133', '133' or the full filename stem all pick the same PDF."""
    squash = lambda s: re.sub(r"\s+", "", s).lower()
    notice = notice_name(pdf.stem)
    return squash(name) in {squash(pdf.stem), squash(notice), notice.split()[-1]}


# ---------- Commands ----------
def dry_run(pdf_files: list[Path]) -> None:
    for pdf in pdf_files:
        chunks = load_pdf(pdf)
        tagged = sum(1 for c in chunks if c.metadata["para_start"])
        print(f"\n{notice_name(pdf.stem)} ({notice_version(pdf.stem)}): {len(chunks)} chunks, "
              f"{tagged} with a paragraph number")
        step = max(1, len(chunks) // 12)
        for c in chunks[::step]:
            print(f"  [{citation(c.metadata)}]  {' '.join(c.page_content.split())[:70]}...")


def ingest(store, pdf_files: list[Path], rebuild: list[str] | None) -> None:
    """rebuild=None: only add new notices. rebuild=[]: redo all. rebuild=[names]: redo those."""
    added = skipped = 0

    for pdf in pdf_files:
        doc, notice = pdf.stem, notice_name(pdf.stem)
        redo = rebuild is not None and (not rebuild or any(matches(pdf, n) for n in rebuild))

        if redo:
            store.delete(where={"notice": notice})   # also clears an older revision of this notice
        elif store.get(where={"doc": doc}, limit=1)["ids"]:
            print(f"Skipping {notice} (already in the store)")
            skipped += 1
            continue
        elif store.get(where={"notice": notice}, limit=1)["ids"]:
            print(f"WARNING: an older revision of {notice} is in the store. "
                  f'Run: uv run python ingest.py --rebuild "{notice}"')
            skipped += 1
            continue

        chunks = load_pdf(pdf)
        print(f"{'Rebuilding' if redo else 'Adding'} {notice} ({notice_version(doc)}): {len(chunks)} chunks")
        if not chunks:
            print("  WARNING: no text found - this PDF is probably scanned and needs OCR")
            continue
        store.add_documents(chunks, ids=chunk_ids(chunks))   # this is the step that calls OpenAI
        added += 1

    unknown = [n for n in rebuild or [] if not any(matches(p, n) for p in pdf_files)]
    if unknown:
        print(f"\nWARNING: no PDF matches {', '.join(unknown)}")
    print(f"\nDone: {added} embedded, {skipped} skipped. "
          f"Store now has {store._collection.count()} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add MAS notice PDFs to the vector store.")
    parser.add_argument("--rebuild", nargs="*", metavar="NOTICE",
                        help='re-embed these notices, e.g. "MAS 133" (no names = every notice)')
    parser.add_argument("--dry-run", action="store_true",
                        help="show chunks and citations without calling OpenAI")
    args = parser.parse_args()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDFs found in {PDF_DIR}")
    if args.dry_run:
        dry_run(pdf_files)
    else:
        load_env()
        ingest(open_vectorstore(), pdf_files, args.rebuild)
