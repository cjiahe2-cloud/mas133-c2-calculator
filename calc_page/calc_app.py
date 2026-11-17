"""Public MAS 133 C2 investment risk calculator: uv run streamlit run calc_page/calc_app.py

No OpenAI key needed: pure Python on top of src/insurance_mas_rag/calculator.py and portfolio.py.
Starts from the sample portfolio (SAMPLE_PORTFOLIO); visitors can change any input.

Lives in its own folder so Streamlit Community Cloud installs calc_page/requirements.txt (streamlit, pandas)
instead of the repo's uv.lock with the chatbot's dependencies. The package isn't installed there, so src/ is
put on the path below.
"""
import base64
import copy
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from insurance_mas_rag.portfolio import (ASSET_CLASSES, FORWARD_RATING, SAMPLE_PORTFOLIO, analyse,  # noqa: E402
                                         validate, weighted_durations)

FROG = str(ROOT / "assets" / "pink_frog.png")
st.set_page_config(page_title="MAS 133 C2 Calculator", page_icon=FROG, layout="wide")

MODULES = [  # key, label, MAS 133 reference
    ("equity", "Equity", "para 4.3.2 · Tables 4D-I/II"),
    ("interest_rate", "Interest rate mismatch", "para 4.3.3.2 · Table 4E"),
    ("credit_spread", "Credit spread", "para 4.3.3.3 · Tables 4F-II/III"),
    ("property", "Property", "para 4.3.4"),
    ("fx", "Foreign currency mismatch", "para 4.3.5"),
    ("default", "Counterparty default", "para 4.3.6 · Table 4H"),
]

st.markdown("""
<style>
.block-container { padding-top: 2rem; max-width: 1150px; }
header[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton, #MainMenu { display: none; }
.hero { display: flex; align-items: center; gap: 1.1rem; padding: 1.3rem 1.5rem; border-radius: 18px;
        background: linear-gradient(120deg, #4F46E5 0%, #635BFF 45%, #E879B9 100%);
        box-shadow: 0 10px 30px -12px rgba(99, 91, 255, .55); margin-bottom: 1.2rem; }
.hero img { width: 70px; height: 70px; border-radius: 50%; background: rgba(255,255,255,.92);
            padding: 6px; flex-shrink: 0; }
.hero-title { color: #fff; font-size: 1.8rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.15; }
.hero-sub { color: rgba(255,255,255,.9); font-size: .95rem; margin-top: .25rem; }
.step { font-size: .74rem; font-weight: 700; color: #635BFF; text-transform: uppercase; letter-spacing: .08em;
        margin: 1.6rem 0 -.2rem; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #E3E8EE; border-radius: 14px; padding: .8rem 1rem;
        box-shadow: 0 1px 3px rgba(10,37,64,.06); }
.status { padding: .75rem 1rem; border-radius: 12px; font-weight: 600; margin: .4rem 0 .8rem; }
.status.ok { background: #ECFDF3; color: #067647; border: 1px solid #ABEFC6; }
.status.bad { background: #FEF3F2; color: #B42318; border: 1px solid #FECDCA; }
.note { font-size: .82rem; color: #425466; line-height: 1.5; }
.disclaimer { font-size: .84rem; color: #425466; line-height: 1.5; background: #F6F9FC; border: 1px solid #E3E8EE;
              border-left: 4px solid #635BFF; border-radius: 10px; padding: .6rem .9rem; margin: -.4rem 0 .4rem; }

/* Inputs: shaded cells, as in a financial model (colour chosen by the user); no +/- steppers */
[data-testid="stNumberInput"] div[data-baseweb="input"], [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #F6F9FC !important; color: #0A2540; }
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stSelectbox"] div[data-baseweb="select"] > div { border-color: #A3B1C2 !important; }
[data-testid="stNumberInput"] button { display: none; }
[data-testid="stNumberInput"] input { text-align: right; font-variant-numeric: tabular-nums; }

/* Asset grid */
.st-key-grid { gap: .35rem; }
.st-key-grid [data-testid="stHorizontalBlock"] { gap: .8rem; }
.grid-head { font-size: .78rem; font-weight: 700; color: #697386; border-bottom: 1px solid #E3E8EE;
             padding-bottom: .35rem; }
.grid-label { font-size: .92rem; color: #0A2540; }
.grid-na { font-size: .8rem; color: #8792A2; text-align: center; }
.grid-total { text-align: right; font-weight: 700; padding: .3rem .75rem; border-top: 2px solid #0A2540;
              font-variant-numeric: tabular-nums; }
.grid-total.bad { color: #B42318; }
.st-key-panel { background: #FBFCFE; border: 1px solid #E3E8EE; border-radius: 16px; padding: 1rem 1.1rem; }
.st-key-panel [data-testid="stMetric"] { padding: .6rem .8rem; }
.st-key-panel [data-testid="stMetricValue"] { font-size: 1.45rem; }
/* Narrow cards: smaller labels on one line, never cut off with "..." */
.st-key-panel [data-testid="stMetric"] { padding: .6rem .55rem .6rem .7rem; }
.st-key-panel [data-testid="stHorizontalBlock"] { gap: .6rem; }
.st-key-panel [data-testid="stMetricLabel"], .st-key-panel [data-testid="stMetricLabel"] * {
        white-space: nowrap !important; overflow: visible !important; text-overflow: clip !important;
        font-size: .76rem !important; }
.panel-gap { margin-top: .5rem; }
.fxbar { display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: #E3E8EE;
         margin: .1rem 0 .45rem; }
.fxlegend { display: flex; flex-wrap: wrap; gap: .3rem .9rem; font-size: .8rem; color: #425466; }
.fxlegend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: .3rem; }
.fxlegend b { color: #0A2540; }
</style>
""", unsafe_allow_html=True)

with open(FROG, "rb") as f:
    frog = base64.b64encode(f.read()).decode()
st.markdown(f"""
<div class="hero"><img src="data:image/png;base64,{frog}" alt="">
  <div><div class="hero-title">MAS 133 C2 Investment Risk Calculator</div>
  <div class="hero-sub">Capital charge for an insurance fund under MAS Notice 133 (Valuation and Capital Framework
  for Insurers). Start from the sample insurer below, then change the allocation, durations and liabilities.</div>
  </div></div>
<div class="disclaimer"><b>Illustrative only.</b> Results depend on the assumptions listed below.</div>""", unsafe_allow_html=True)

# ---------- Inputs ----------
if "v" not in st.session_state:
    st.session_state.v = 0          # bumped by "Reset" so every input starts again from the sample
v = st.session_state.v
S = SAMPLE_PORTFOLIO

head, reset = st.columns([5, 1], vertical_alignment="bottom")
head.markdown('<div class="step">1 · Portfolio</div>', unsafe_allow_html=True)
if reset.button("Reset to sample", icon=":material/restart_alt:", width="stretch"):
    st.session_state.v += 1
    st.rerun()

c1, c2, c3, c4 = st.columns(4)
total_assets = c1.number_input("Total assets (SGD m)", 10.0, 1e6, S["total_assets"], 50.0, key=f"ta{v}")
base_yield = c2.number_input("Base yield (%)", 0.0, 15.0, S["base_yield"] * 100, 0.25, key=f"y{v}") / 100
liab_value = c3.number_input("Liabilities (SGD m)", 0.0, 1e6, S["liabilities"]["value"], 50.0, key=f"lv{v}")
liab_dur = c4.number_input("Liability duration (years)", 0.0, 50.0, S["liabilities"]["duration"], 1.0,
                           key=f"ld{v}")

# Asset grid: one row per asset class, shaded input cells like a financial model (an editable st.data_editor
# can't colour its editable cells, so each cell is its own number_input).
GRID = [3.2, 1.7, 1.7, 1.6]
FIXED_RATING = {"sgs": "AAA", "ust": "A or better", "unrated": "Unrated"}   # shown, not editable
alloc, duration, ratings = {}, {}, {}
left, right = st.columns([7, 4], gap="large")
with left, st.container(key="grid"):
    for col, text in zip(st.columns(GRID), ["Asset class", "Allocation (% of fund)", "Effective duration (years)",
                                            "Credit rating"]):
        col.markdown(f'<div class="grid-head">{text}</div>', unsafe_allow_html=True)
    for k, c in ASSET_CLASSES.items():
        row = st.columns(GRID, vertical_alignment="center")
        row[0].markdown(f'<div class="grid-label">{c["label"]}</div>', unsafe_allow_html=True)
        alloc[k] = row[1].number_input(f"{c['label']} allocation", 0.0, 100.0, S["allocation"][k] * 100, 0.5,
                                       format="%.1f", key=f"a_{k}{v}", label_visibility="collapsed") / 100
        if c["kind"] == "bond":
            duration[k] = row[2].number_input(f"{c['label']} duration", 0.0, 40.0, S["duration"][k], 0.5,
                                              format="%.1f", key=f"d_{k}{v}", label_visibility="collapsed")
        else:
            row[2].markdown('<div class="grid-na">–</div>', unsafe_allow_html=True)
        if "ratings" in c:
            ratings[k] = row[3].selectbox(f"{c['label']} rating", c["ratings"],
                                          index=c["ratings"].index(S["rating"][k]), key=f"cr_{k}{v}",
                                          label_visibility="collapsed")
        else:
            row[3].markdown(f'<div class="grid-na">{FIXED_RATING.get(k, "–")}</div>', unsafe_allow_html=True)
    total = sum(alloc.values())
    row = st.columns(GRID, vertical_alignment="center")
    row[0].markdown('<div class="grid-label"><b>Total</b></div>', unsafe_allow_html=True)
    row[1].markdown(f'<div class="grid-total {"ok" if abs(total - 1) < 1e-6 else "bad"}">{total * 100:.1f}</div>',
                    unsafe_allow_html=True)
    row[2].markdown('<div class="grid-na">Shaded cells are inputs</div>', unsafe_allow_html=True)
bond_dur, port_dur = weighted_durations(alloc, duration)
bonds_pct = sum(alloc[k] for k, c in ASSET_CLASSES.items() if c["kind"] == "bond")
listed_pct = sum(alloc[k] for k, c in ASSET_CLASSES.items() if c["kind"] == "equity" and not c.get("unlisted"))

# Summary panel beside the grid: durations, mix, FX hedge inputs
with right, st.container(key="panel"):
    st.markdown('<div class="grid-head">Portfolio summary</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    s1.metric("Bond duration", f"{bond_dur:.1f} yrs" if bond_dur is not None else "–",
              help="Market-value weighted average effective duration of the bonds.")
    s2.metric("Portfolio duration", f"{port_dur:.1f} yrs",
              help="Market-value weighted average effective duration of all assets; equity and property count "
                   "as zero.")
    st.metric("Bonds / listed equities / others",
              f"{bonds_pct:.0%} / {listed_pct:.0%} / {1 - bonds_pct - listed_pct:.0%}",
              help="Listed equities are developed- and emerging-market listed equity. Others are private equity "
                   "and property.")

    # Currency mix: SGD % and FX forwards % are inputs; non-SGD and natural SGD follow from them
    st.markdown('<div class="grid-head panel-gap">Currency mix (% of fund)</div>', unsafe_allow_html=True)
    x1, x2 = st.columns(2, vertical_alignment="bottom")
    sgd_share = x1.number_input("SGD %", 0.0, 100.0, S["sgd_share"] * 100, 5.0, key=f"sgd{v}",
                                help="SGD exposure after hedging: natural SGD assets plus FX forwards.") / 100
    hedged = x2.number_input("FX forwards %", 0.0, 100.0, S["hedged"] * 100, 5.0, key=f"h{v}",
                             help="Notional of forwards selling USD for SGD, as a % of the fund's market value. "
                                  "Part of the SGD share.") / 100
    natural, non_sgd = max(sgd_share - hedged, 0.0), 1 - sgd_share
    st.markdown(f"""
<div class="fxbar"><span style="width:{natural:.2%};background:#0A2540"></span><span
  style="width:{min(hedged, sgd_share):.2%};background:#635BFF"></span><span
  style="width:{non_sgd:.2%};background:#E879B9"></span></div>
<div class="fxlegend"><span><i style="background:#0A2540"></i>Natural SGD <b>{natural:.0%}</b></span>
  <span><i style="background:#635BFF"></i>FX forwards <b>{hedged:.0%}</b></span>
  <span><i style="background:#E879B9"></i>Non-SGD (USD) <b>{non_sgd:.0%}</b></span></div>""",
                unsafe_allow_html=True)
    f1, f2 = st.columns(2, vertical_alignment="bottom")
    months = f1.selectbox("Forward maturity", [1, 3, 6, 12, 24], index=2, format_func=lambda x: f"{x} months",
                          key=f"m{v}")
    rating = f2.selectbox("Counterparty rating", list(FORWARD_RATING),
                          index=list(FORWARD_RATING).index(S["forward_rating"]), key=f"r{v}")

portfolio = copy.deepcopy(S) | {
    "total_assets": total_assets, "base_yield": base_yield, "allocation": alloc, "duration": duration, "rating": ratings,
    "liabilities": {"value": liab_value, "duration": liab_dur},
    "sgd_share": sgd_share, "hedged": hedged, "forward_months": months, "forward_rating": rating}

errors = validate(portfolio)
if errors:
    for e in errors:
        st.error(e, icon=":material/error:")
    st.stop()

a = analyse(portfolio)
c2 = a.c2

# ---------- Breakdown ----------
st.markdown('<div class="step">2 · C2 risk charge breakdown</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("C2 investment risk requirement", f"{c2.c2:,.1f}m", f"{c2.c2 / total_assets:.1%} of assets",
          delta_color="off")
m2.metric("Sum of standalone charges", f"{c2.undiversified:,.1f}m")
m3.metric("Diversification benefit", f"{c2.undiversified - c2.c2:,.1f}m",
          f"{(c2.undiversified - c2.c2) / c2.undiversified:.0%} of standalone" if c2.undiversified else None,
          delta_color="off")
m4.metric("Interest rate scenario", "Rates down" if a.scenario == "down" else "Rates up",
          f"Table {'4C-II' if a.scenario == 'down' else '4C-I'} correlations", delta_color="off")

rows = [{"Risk module": label, "Standalone (SGD m)": a.modules[k], "% of assets": a.modules[k] / total_assets * 100,
         "Contribution to C2 (SGD m)": c2.contributions[k],
         "Share of C2": c2.contributions[k] / c2.c2 * 100 if c2.c2 else 0.0, "MAS 133": ref}
        for k, label, ref in MODULES]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
    "Standalone (SGD m)": st.column_config.NumberColumn(format="%.2f"),
    "% of assets": st.column_config.NumberColumn(format="%.2f%%"),
    "Contribution to C2 (SGD m)": st.column_config.NumberColumn(
        format="%.2f", help="Euler allocation: each module's share of the diversified C2. Adds up to C2."),
    "Share of C2": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
st.markdown(f'<div class="note">Modules are combined with the Table '
            f'{"4C-II (downward)" if a.scenario == "down" else "4C-I (upward)"} correlation matrix, chosen by the '
            f'interest rate scenario that gives the larger loss. Market risk {c2.market:,.2f}m is then combined with '
            f'counterparty default at correlation 0.5 (para 4.3.1.1).</div>', unsafe_allow_html=True)

with st.expander("Show the working for each module"):
    for k, label, ref in MODULES:
        st.markdown(f"**{label}** · {ref}")
        st.code("\n".join(line.strip() for line in a.steps[k]) or "No exposure.", language=None, wrap_lines=True)
    st.markdown("**Aggregation** · paras 4.3.1.1, 4.3.1.4")
    st.code("\n".join(line.strip() for line in c2.lines), language=None, wrap_lines=True)

# ---------- Concentration limits ----------
st.markdown('<div class="step">3 · Concentration limits (Appendix 5E, Table 2-I)</div>', unsafe_allow_html=True)
breaches = [r for r in a.limits if r.breach]
if breaches:
    st.markdown(f'<div class="status bad">✕ {len(breaches)} limit{"s" if len(breaches) > 1 else ""} breached. '
                f'Adjustment for asset concentration: {a.concentration_adjustment:,.2f}m, deducted from financial '
                f'resources (para 5.1(a)(ii)).</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status ok">✓ All concentration limits are met. No adjustment to financial '
                'resources.</div>', unsafe_allow_html=True)

st.dataframe(pd.DataFrame([{
    "Item": r.item, "Limit applies to": r.covers[0].upper() + r.covers[1:], "Checked against": r.holding,
    "Exposure (SGD m)": r.exposure, "Limit (SGD m)": r.limit,
    "Limit used": min(r.exposure / r.limit * 100, 100) if r.limit else 0.0,
    "Status": "✕ Breach" if r.breach else "✓ Within", "Adjustment (SGD m)": r.adjustment} for r in a.limits]),
    hide_index=True, width="stretch", column_config={
        "Exposure (SGD m)": st.column_config.NumberColumn(format="%.2f"),
        "Limit (SGD m)": st.column_config.NumberColumn(format="%.2f"),
        "Limit used": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Adjustment (SGD m)": st.column_config.NumberColumn(format="%.2f")})
if breaches:
    st.markdown('<div class="note">Para 4.1.8 also excludes the assets behind the adjustment from the C2 '
                'calculation. Apart from the FX module, this page does not recalculate C2 for that.</div>',
                unsafe_allow_html=True)
with st.expander("Show the FX and concentration working"):
    st.code("\n".join(line.strip() for line in a.fx_lines), language=None, wrap_lines=True)

# ---------- Assumptions ----------
st.markdown('<div class="step">Assumptions</div>', unsafe_allow_html=True)
st.markdown(f"""<div class="note">
• Amounts in SGD millions. One insurance fund for Singapore policies, with no reinsurers' share or unit reserves.<br>
• The currency mix is entered directly rather than taken from the asset classes: the SGD share is natural SGD
assets plus FX forwards selling USD for SGD, and the rest of the fund is in USD. Liabilities are in SGD.<br>
• Interest rate and credit spread charges use the duration approach (para 4.3.3.2 c)(i)), one base yield for every
asset and liability, and the Table 4E row closest to the duration rather than each cash flow's term.<br>
• Government bonds are rated A or better, so they have no credit spread charge (para 4.3.3.3 i)).<br>
• Every exposure to a single issuer or counterparty (one company, one equity, one private equity holding) is
within its Table 2-I limit, so only the aggregate limits (private equity, property, FX) and the FX forward bank
are checked.<br>• The only counterparty default exposure is the FX forwards (credit equivalent amount, Appendix 4H). Equity
cyclical adjustments (CCA) are zero.<br>
• This is an illustration of the MAS 133 method, not regulatory or professional advice. Where the Notice leaves
room for interpretation, the page follows one reading; check it against MAS Notice 133 and your insurer's own
methodology.
</div>""", unsafe_allow_html=True)
