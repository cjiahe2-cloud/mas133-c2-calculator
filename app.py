import uuid

import streamlit as st

from insurance_mas_rag import CHAT_MODEL, Context, build_agent, latest_turn, load_env

st.set_page_config(page_title="MAS Notices Assistant", page_icon="⚖️")

try:
    load_env()
except SystemExit as e:
    st.error(str(e))
    st.stop()

# Build the agent once (Streamlit reruns this file on every click)
agent = st.cache_resource(build_agent)()


def safe_md(text: str) -> str:
    """Streamlit treats $...$ as maths; escape so '$100' displays normally."""
    return text.replace("$", "\\$")


# ---------- Page setup ----------
st.title("⚖️ MAS Notices Assistant")
st.caption(f"Answers come only from MAS Notice 125 (30 Sep 2022) and MAS Notice 133 (16 Mar 2026) · "
           f"not legal advice · model: {CHAT_MODEL}")

# Each browser session gets its own conversation memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []   # [{role, content, searches, passages}]

with st.sidebar:
    st.header("Session")
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.markdown("**Try asking:**\n- Who must sit on the Investment Committee?\n"
                "- What must an ALM policy include?\n"
                "- What are the supervisory solvency intervention levels?\n"
                "- What is the claim liability risk factor for medium volatility?")

# ---------- Show previous messages ----------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(safe_md(msg["content"]))
        if msg.get("searches"):
            with st.expander(f"🔎 Searched: {'; '.join(msg['searches'])}"):
                for p in msg["passages"]:
                    st.text(p)

# ---------- New question ----------
if question := st.chat_input("Ask about MAS 125 or MAS 133..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(safe_md(question))

    with st.chat_message("assistant"):
        with st.spinner("Searching the notices..."):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
                context=Context(user_id="me"),
            )

        reply, searches, passages = latest_turn(result["messages"])

        st.markdown(safe_md(reply))
        if searches:
            with st.expander(f"🔎 Searched: {'; '.join(searches)}"):
                for p in passages:
                    st.text(p)

    st.session_state.history.append(
        {"role": "assistant", "content": reply, "searches": searches, "passages": passages}
    )
