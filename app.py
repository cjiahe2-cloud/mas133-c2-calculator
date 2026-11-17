import base64
import html
import re
import uuid

import streamlit as st

from insurance_mas_rag.rag import CALC_HEADER, CALC_PASSAGES, CHAT_MODEL, PROJECT_DIR, Context, build_agent, latest_turn, load_env

FROG = str(PROJECT_DIR / "assets" / "pink_frog.png")
USER_ICON = ":material/person:"
st.set_page_config(page_title="MAS 133 & MAS 125", page_icon=FROG)

try:
    load_env()
except SystemExit as e:
    st.error(str(e))
    st.stop()

# Build the agent once (Streamlit reruns this file on every click)
agent = st.cache_resource(build_agent)(CHAT_MODEL)

NOTICES = [("MAS 125", "Investments of Insurers", "30 Sep 2022"),
           ("MAS 133", "Valuation and Capital Framework", "16 Mar 2026")]

STARTERS = [("🏛️", "Investment Committee", "Who must sit on an insurer's Investment Committee?"),
            ("⚖️", "Asset-liability management", "What must an ALM policy include?"),
            ("🛡️", "Solvency intervention levels", "What are the supervisory solvency intervention levels?"),
            ("🧩", "Risk grouping (HRGs)", "How should an insurer group policies into HRGs?")]

# ---------- Styling ----------
# Colours follow .streamlit/config.toml (white, pale grey-blue, blurple #635BFF, navy #0A2540)
st.markdown("""
<style>
/* Layout and Streamlit chrome */
.block-container { padding-top: 2.2rem; max-width: 820px; }
header[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton, #MainMenu { display: none; }

/* 1. Header banner */
.hero { display: flex; align-items: center; gap: 1.1rem; padding: 1.3rem 1.5rem; border-radius: 18px;
        background: linear-gradient(120deg, #4F46E5 0%, #635BFF 45%, #E879B9 100%);
        box-shadow: 0 10px 30px -12px rgba(99, 91, 255, .55); margin-bottom: 1.4rem; }
.hero img { width: 74px; height: 74px; border-radius: 50%; background: rgba(255,255,255,.92);
            padding: 6px; box-shadow: 0 4px 14px rgba(10,37,64,.18); flex-shrink: 0; }
.hero-title { color: #fff; font-size: 1.85rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.15; }
.hero-sub { color: rgba(255,255,255,.88); font-size: .95rem; margin-top: .2rem; }
.hero-meta { margin-top: .6rem; display: flex; gap: .45rem; flex-wrap: wrap; }
.hero-pill { font-size: .74rem; font-weight: 600; color: #fff; background: rgba(255,255,255,.18);
             border: 1px solid rgba(255,255,255,.35); padding: .18rem .65rem; border-radius: 999px; }

/* 2. Welcome cards */
.welcome-title { font-weight: 600; color: #0A2540; margin: .3rem 0 .1rem; }
.welcome-sub { color: #425466; font-size: .9rem; margin-bottom: .6rem; }
.st-key-starters button { height: 100%; min-height: 96px; text-align: left; justify-content: flex-start;
        align-items: flex-start; padding: .9rem 1rem; border-radius: 14px; border: 1px solid #E3E8EE;
        background: #fff; box-shadow: 0 1px 3px rgba(10,37,64,.06); transition: all .15s ease; }
.st-key-starters button:hover { border-color: #635BFF; box-shadow: 0 8px 22px -10px rgba(99,91,255,.45);
        transform: translateY(-2px); color: #0A2540; }
.st-key-starters button > div, .st-key-starters button [data-testid="stMarkdownContainer"] {
        width: 100%; justify-content: flex-start; text-align: left; }
.st-key-starters button p { text-align: left; font-size: .9rem; line-height: 1.45; white-space: normal !important;
        overflow: visible !important; text-overflow: clip !important; }
.st-key-notice_filter button { padding: .25rem .7rem; font-size: .82rem; min-height: 2rem; }
.st-key-notice_filter [data-testid="stButtonGroup"] > div { flex-wrap: nowrap; }

/* 6. Chat messages */
[data-testid="stChatMessage"] { background: #fff; border: 1px solid #E3E8EE; border-radius: 16px;
        padding: 1rem 1.1rem; margin-bottom: .8rem; box-shadow: 0 2px 10px -4px rgba(10,37,64,.10); }
[data-testid="stChatInput"] { border-radius: 14px; box-shadow: 0 6px 20px -10px rgba(10,37,64,.25); }

/* 3. Requirement / Guideline / Information badges */
.badge { display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .02em;
         padding: .12rem .55rem; border-radius: 999px; margin-right: .35rem; vertical-align: .08em; }
.badge-req  { background: #FDECEC; color: #B42318; border: 1px solid #F9C9C5; }
.badge-guid { background: #FEF3C7; color: #92400E; border: 1px solid #FCDFA0; }
.badge-info { background: #EEF2F6; color: #475467; border: 1px solid #DCE3EB; }

/* 4. Citation chips and sources panel */
.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .7rem 0 .3rem; }
.chips-label { font-size: .72rem; font-weight: 700; color: #697386; text-transform: uppercase;
               letter-spacing: .06em; align-self: center; margin-right: .1rem; }
.chip { font-size: .76rem; font-weight: 500; color: #4338CA; background: #EEF0FF;
        border: 1px solid #D9DBFF; padding: .16rem .6rem; border-radius: 999px; }
.src-card { border: 1px solid #E3E8EE; border-left: 3px solid #635BFF; border-radius: 10px;
            padding: .6rem .8rem; margin-bottom: .55rem; background: #fff; }
.src-label { font-size: .78rem; font-weight: 700; color: #4338CA; margin-bottom: .25rem; }
.src-text { font-size: .82rem; color: #425466; line-height: 1.45; }
.src-searches { font-size: .78rem; color: #697386; margin-bottom: .6rem; }

/* 5. Sidebar */
[data-testid="stSidebar"] .side-card { background: #fff; border: 1px solid #E3E8EE; border-radius: 12px;
        padding: .75rem .85rem; margin-top: .9rem; font-size: .82rem; color: #425466; }
.side-card b { color: #0A2540; }
.side-head { font-size: .72rem; font-weight: 700; color: #697386; text-transform: uppercase;
             letter-spacing: .06em; margin-bottom: .45rem; }
.side-row { margin-bottom: .45rem; line-height: 1.35; }
.side-date { color: #697386; font-size: .76rem; }
.side-card.warn { background: #FFFBEB; border-color: #FCDFA0; color: #78350F; }
</style>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
def safe_md(text: str) -> str:
    """Escape HTML (replies are shown with unsafe_allow_html for the badges) and '$', which
    Streamlit treats as maths, so '$100' displays normally."""
    return html.escape(text, quote=False).replace("$", "\\$")


BADGES = [(r"\*\*Requirement(?: \(must\))?:?\*\*:?", '<span class="badge badge-req">REQUIREMENT · MUST</span>'),
          (r"\*\*Guideline(?: \(should\))?:?\*\*:?", '<span class="badge badge-guid">GUIDELINE · SHOULD</span>'),
          (r"\*\*Information:?\*\*:?", '<span class="badge badge-info">INFORMATION</span>')]


def parse_passages(tool_outputs: list[str]) -> list[tuple[str, str]]:
    """Split search results into (citation label, text), dropping repeats across searches."""
    seen, out = set(), []
    for content in tool_outputs:
        parts = re.split(r"(?m)^\[(MAS \d+ p\.[^\]\n]*)\]\n", content)
        for label, text in zip(parts[1::2], parts[2::2]):
            key = (label, text[:80])
            if key not in seen:
                seen.add(key)
                out.append((label, text.strip()))
    return out


def chip_text(label: str) -> str:
    """'MAS 133 p.31, para 4.2.13' -> 'MAS 133 · p.31 · para 4.2.13'."""
    return re.sub(r" (p\.)", r" · \1", label).replace(", ", " · ")


def render_answer(reply: str, searches: list[str], passages: list[str]) -> None:
    sources = parse_passages(passages)
    # Cited sources: retrieved passages whose notice and page appear in the reply
    cited = []
    for label, _ in sources:
        page = re.match(r"MAS \d+ p\.\d+", label).group()
        if re.search(re.escape(page) + r"\b", reply) and label not in cited:
            cited.append(label)

    body = reply
    if cited:   # the chips replace the trailing 'Sources:' line
        heads = list(re.finditer(r"(?m)^\s*(?:\*\*)?Sources?(?:\*\*)?\s*:", body))
        if heads and heads[-1].start() > len(body) // 3:
            body = body[:heads[-1].start()].rstrip()

    body = safe_md(body)
    for pattern, badge in BADGES:
        body = re.sub(pattern, badge, body)
    st.markdown(body, unsafe_allow_html=True)

    if cited:
        chips = "".join(f'<span class="chip">{html.escape(chip_text(c))}</span>' for c in cited)
        st.markdown(f'<div class="chips"><span class="chips-label">Sources</span>{chips}</div>',
                    unsafe_allow_html=True)

    for calc in (p for p in passages if p.startswith(CALC_HEADER)):
        with st.expander("🧮 Calculation steps (MAS 133 calculator)"):
            st.code(calc.split("\n", 1)[1].split(CALC_PASSAGES)[0].strip(), language=None, wrap_lines=True)

    if sources:
        with st.expander(f"🔎 {len(sources)} passages searched"):
            if searches:
                st.markdown(f'<div class="src-searches">Searches: {html.escape(" · ".join(searches))}</div>',
                            unsafe_allow_html=True)
            for label, text in sources:
                snippet = " ".join(text.split())
                snippet = snippet[:420] + ("…" if len(snippet) > 420 else "")
                st.markdown(f'<div class="src-card"><div class="src-label">{html.escape(chip_text(label))}</div>'
                            f'<div class="src-text">{html.escape(snippet)}</div></div>',
                            unsafe_allow_html=True)


# ---------- Session ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []   # [{role, content, searches, passages}]


def ask(question: str) -> None:
    st.session_state.pending = question


# ---------- 5. Sidebar ----------
with st.sidebar:
    st.markdown('<div class="side-head">Search in</div>', unsafe_allow_html=True)
    choice = st.segmented_control("Search in", ["Both", "MAS 125", "MAS 133"], default="Both",
                                  key="notice_filter", label_visibility="collapsed")
    notice = None if choice in (None, "Both") else choice

    if st.button("New conversation", icon="✨", type="primary", width="stretch"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    rows = "".join(f'<div class="side-row"><b>{n}</b> · {title}<br><span class="side-date">Revised {date}</span></div>'
                   for n, title, date in NOTICES)
    st.markdown(f'<div class="side-card"><div class="side-head">📚 Sources</div>{rows}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-card warn"><b>Not legal advice.</b> Information about the notices only. '
                'Answers can be wrong: check the quoted paragraph in the notice.</div>', unsafe_allow_html=True)

# ---------- 1. Header ----------
with open(FROG, "rb") as f:
    frog_b64 = base64.b64encode(f.read()).decode()
st.markdown(f"""
<div class="hero">
  <img src="data:image/png;base64,{frog_b64}" alt="">
  <div>
    <div class="hero-title">MAS 133 &amp; MAS 125</div>
    <div class="hero-sub">Ask about MAS insurance notices. Every answer is quoted and cited.</div>
    <div class="hero-meta"><span class="hero-pill">Not legal advice</span>
      <span class="hero-pill">Model: {html.escape(CHAT_MODEL)}</span></div>
  </div>
</div>""", unsafe_allow_html=True)

# ---------- Conversation ----------
question = st.chat_input("Ask about MAS 125 or MAS 133...")
if not question:
    question = st.session_state.pop("pending", None)

for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user", avatar=USER_ICON):
            st.markdown(safe_md(msg["content"]))
    else:
        with st.chat_message("assistant", avatar=FROG):
            render_answer(msg["content"], msg.get("searches", []), msg.get("passages", []))

# ---------- 2. Welcome cards (before the first question) ----------
if not st.session_state.history and not question:
    st.markdown('<div class="welcome-title">Start with a question</div>'
                '<div class="welcome-sub">Pick one below or type your own.</div>', unsafe_allow_html=True)
    with st.container(key="starters"):
        cols = st.columns(2)
        for i, (icon, title, q) in enumerate(STARTERS):
            cols[i % 2].button(f"{icon} **{title}**  \n{q}", key=f"starter{i}", width="stretch",
                               on_click=ask, args=(q,))

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_ICON):
        st.markdown(safe_md(question))

    with st.chat_message("assistant", avatar=FROG):
        with st.spinner("Searching the notices..."):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
                context=Context(user_id="me", notice=notice),
            )
        reply, searches, passages = latest_turn(result["messages"])
        render_answer(reply, searches, passages)

    st.session_state.history.append(
        {"role": "assistant", "content": reply, "searches": searches, "passages": passages}
    )
