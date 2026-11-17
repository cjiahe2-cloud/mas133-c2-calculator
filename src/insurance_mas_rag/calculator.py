"""Deterministic MAS 133 calculations the chat model kept getting wrong when it chained the rules itself:

- foreign currency mismatch risk requirement (paras 4.3.5.3-4.3.5.7)
- adjustment for asset concentration (Appendix 5E paras 16-17B, Table 2-I), deducted from financial
  resources (para 5.1(a)(ii))
- the FX requirement recalculated after excluding the assets that cause the concentration adjustment
  from the C2 calculation (para 4.1.8)
- the other C2 modules and their aggregation (second half of the file): equity (4.3.2), interest rate mismatch
  and credit spread by the duration approach or by discounted cash flows (4.3.3), property (4.3.4), counterparty default (4.3.6),
  miscellaneous (4.3.7) and the diversified C2 (4.3.1.1, 4.3.1.4) with an Euler split per module

All amounts are SGD millions. How the FX part of the adjustment is worked out is set by FX_ADJUSTMENT_METHOD
below (the user's reading: total asset value less the 40% limit). The interpretation of para 4.1.8 (which assets are excluded, and whether the
10% / 20% deduction base shrinks with them) is our reading of the text: it should be confirmed by an actuary.
"""
import math
from dataclasses import dataclass, field

MIN_LIMIT = 5.0          # para 17B: a limit of $5 million applies where the % limit gives less
FX_FACTOR = 0.12         # para 4.3.5.3(a)
FX_DEDUCTION = {"singapore": 0.10, "offshore": 0.20}            # paras 4.3.5.6 / 4.3.5.7
FX_PARA = {"singapore": "4.3.5.6", "offshore": "4.3.5.7"}
FX_LIMIT = 0.40          # Table 2-I item (5): only FX exposure under para 4.3.5.6 (Singapore policies)

# How the FX part of the adjustment for asset concentration (Appendix 5E para 16) is worked out, once the
# FX risk exposure is over the 40% limit:
#   "total_less_limit": (a) total asset value less (b) 40% x total asset value
#                       e.g. 100m fund, 60% USD: 100m - 40m = 60m. Para 4.1.8 recalculation not applied.
#   "excess":           only the FX risk exposure above the limit, e.g. 100m fund, 60% USD: exposure
#                       60m - 10m = 50m, less 40m = 10m; then para 4.1.8 recalculates the FX requirement.
FX_ADJUSTMENT_METHOD = "total_less_limit"

# Table 2-I items 1-4 and 6 (item 5, FX, is calculated from the currency positions). % of total assets
# excluding reinsurers' share of policy liabilities (and unit reserves of an investment-linked fund, para 16A).
LIMITS = {
    "counterparty_government": (1.00, "(1)(a)", "counterparty: Government, investment-grade sovereign or "
                                                "central bank, Government-owned company or statutory board"),
    "counterparty_approved_fi": (0.20, "(1)(b)", "counterparty: approved financial institution"),
    "counterparty_listed": (0.10, "(1)(c)", "counterparty: other entity, listed"),
    "counterparty_unlisted": (0.05, "(1)(d)", "counterparty: other entity, not listed"),
    "equity_listed_single": (0.05, "(2)(a)", "one listed equity security (not a CIS)"),
    "equity_unlisted_single": (0.025, "(2)(b)", "one unlisted equity (not a CIS)"),
    "equity_unlisted_aggregate": (0.10, "(2)(c)", "unlisted equities in aggregate (not CIS)"),
    "equity_cis": (0.10, "(2)(d)", "a collective investment scheme"),
    "unsecured_loan_single": (0.01, "(3)(a)", "unsecured loans to a single counterparty"),
    "unsecured_loans_aggregate": (0.025, "(3)(b)", "unsecured loans in aggregate"),
    "property": (0.35, "(4)", "property exposure"),
    "miscellaneous_aggregate": (0.025, "(6)", "assets subject to the miscellaneous risk factor (para 4.3.7.1)"),
}


@dataclass
class Result:
    lines: list[str] = field(default_factory=list)
    fx_exposure: float = 0.0
    fx_requirement: float = 0.0
    fx_requirement_after_exclusion: float = 0.0
    concentration_adjustment: float = 0.0

    def add(self, text: str = "") -> None:
        self.lines.append(text)


def m(x: float) -> str:
    return f"{x:,.2f}m".replace(".00m", "m")


def pct(x: float, base: float) -> str:
    return f"{x / base:.2%}" if base else "n/a"


def limit_amount(rate: float, base: float) -> tuple[float, str]:
    raw = rate * base
    if raw < MIN_LIMIT:
        return MIN_LIMIT, f"{rate:.1%} x {m(base)} = {m(raw)}, below $5m, so the para 17B minimum of 5m applies"
    return raw, f"{rate:.1%} x {m(base)} = {m(raw)}"


def calculate(fund_type: str, total_assets: float, fx_positions: dict[str, float] | None = None,
              reinsurers_share: float = 0.0, unit_reserves: float = 0.0,
              exposures: list[dict] | None = None, method: str | None = None) -> Result:
    """fx_positions: net open position per foreign currency in SGD m, signed (assets less liabilities plus
    hedges, para 4.3.5.4); exposures: [{"item": key of LIMITS, "amount": SGD m, "name": optional label}];
    method: FX adjustment method (default FX_ADJUSTMENT_METHOD)."""
    fund_type = fund_type.lower()
    method = method or FX_ADJUSTMENT_METHOD
    if fund_type not in FX_DEDUCTION:
        raise ValueError("fund_type must be 'singapore' or 'offshore'")
    if method not in ("total_less_limit", "excess"):
        raise ValueError("method must be 'total_less_limit' or 'excess'")
    r = Result()
    fx = {k.upper(): v for k, v in (fx_positions or {}).items() if k.upper() != "SGD"}
    ta_net = total_assets - reinsurers_share                 # para 4.3.5.6/7 deduction base
    conc_base = ta_net - unit_reserves                        # Table 2-I base (para 16A)

    # ---- 1. FX mismatch risk requirement ----
    para, rate = FX_PARA[fund_type], FX_DEDUCTION[fund_type]
    pos = sum(v for v in fx.values() if v > 0)
    neg = abs(sum(v for v in fx.values() if v < 0))
    gross = max(pos, neg)
    deduction = rate * ta_net
    r.fx_exposure = max(0.0, gross - deduction)
    r.fx_requirement = FX_FACTOR * r.fx_exposure
    r.add(f"1. Foreign currency mismatch risk requirement (fund for {fund_type} policies, para {para})")
    if fx:
        r.add("   Net open positions (SGD excluded, para 4.3.5.4): "
              + ", ".join(f"{c} {m(v)}" for c, v in fx.items()))
    else:
        r.add("   No foreign currency positions given.")
    r.add(f"   Positive positions {m(pos)}, negative positions {m(neg)}; higher = {m(gross)}")
    r.add(f"   Less {rate:.0%} of (total assets {m(total_assets)} less reinsurers' share {m(reinsurers_share)})"
          f" = {m(deduction)}")
    r.add(f"   FX risk exposure = max(0, {m(gross)} - {m(deduction)}) = {m(r.fx_exposure)}")
    r.add(f"   Requirement = 12% x {m(r.fx_exposure)} = {m(r.fx_requirement)} "
          f"({pct(r.fx_requirement, total_assets)} of total assets) (para 4.3.5.3)")

    # ---- 2. Adjustment for asset concentration ----
    r.add("")
    r.add("2. Adjustment for asset concentration (Appendix 5E paras 16-17B, Table 2-I)")
    r.add(f"   Limit base: total assets {m(total_assets)} less reinsurers' share {m(reinsurers_share)}"
          + (f" less unit reserves {m(unit_reserves)} (para 16A)" if unit_reserves else "")
          + f" = {m(conc_base)}")
    excesses = []
    fx_excess = fx_adjustment = 0.0
    if fund_type == "singapore":
        lim, how = limit_amount(FX_LIMIT, conc_base)
        fx_excess = max(0.0, r.fx_exposure - lim)
        if method == "total_less_limit":
            fx_adjustment = conc_base - lim if fx_excess else 0.0
            how += (f"; exposure is over the limit, so FX adjustment = (a) total asset value {m(conc_base)} less "
                    f"(b) {m(lim)} = {m(fx_adjustment)}" if fx_excess else "; exposure is within the limit")
        else:
            fx_adjustment = fx_excess
        excesses.append(("item (5) FX risk exposure under para 4.3.5.6", r.fx_exposure, how, fx_adjustment))
    else:
        r.add("   Item (5), the 40% FX limit, covers only FX exposure under para 4.3.5.6 (Singapore policies), "
              "so it does not apply to this offshore fund.")
    for e in exposures or []:
        rate_i, item, desc = LIMITS[e["item"]]
        lim, how = limit_amount(rate_i, conc_base)
        label = f"item {item} {desc}" + (f" - {e['name']}" if e.get("name") else "")
        excesses.append((label, e["amount"], how, max(0.0, e["amount"] - lim)))
    for label, amount, how, excess in excesses:
        r.add(f"   {label}: exposure {m(amount)}, limit {how}; adjustment {m(excess)}")
    r.concentration_adjustment = sum(x[3] for x in excesses)
    r.add(f"   Total adjustment for asset concentration = {m(r.concentration_adjustment)} "
          f"({pct(r.concentration_adjustment, total_assets)} of total assets)")
    r.add("   It is deducted from financial resources (para 5.1(a)(ii)); it is not a risk requirement.")
    if len(excesses) > 1:
        r.add("   Check: an asset that breaches two limits is only excess once, so overlapping excesses must not "
              "be added twice.")

    # ---- 3. Para 4.1.8: exclude assets causing the adjustment from C2 ----
    r.add("")
    r.add("3. FX requirement after excluding concentration excess from C2 (para 4.1.8)")
    if fx_excess and method == "total_less_limit":
        r.fx_requirement_after_exclusion = r.fx_requirement
        r.add("   Not recalculated: which assets para 4.1.8 excludes from C2 under this adjustment method has not "
              "been settled. The FX requirement above is before any para 4.1.8 exclusion; confirm with an "
              "actuary.")
    elif fx_excess and pos >= neg:
        pos2 = pos - fx_excess
        gross2 = max(pos2, neg)
        exp_a = max(0.0, gross2 - deduction)
        exp_b = max(0.0, gross2 - rate * (ta_net - fx_excess))
        r.fx_requirement_after_exclusion = FX_FACTOR * exp_a
        r.add(f"   Positive positions {m(pos)} less excess {m(fx_excess)} = {m(pos2)}; higher of that and "
              f"{m(neg)} = {m(gross2)}")
        r.add(f"   FX exposure = {m(gross2)} - {m(deduction)} = {m(exp_a)}; requirement = 12% x {m(exp_a)} = "
              f"{m(FX_FACTOR * exp_a)} ({pct(FX_FACTOR * exp_a, total_assets)} of total assets)")
        r.add(f"   Assumption: the {rate:.0%} deduction stays on the fund's total assets. If it is taken on the "
              f"assets left after the exclusion, exposure = {m(gross2)} - {m(rate * (ta_net - fx_excess))} = "
              f"{m(exp_b)} and requirement = {m(FX_FACTOR * exp_b)} "
              f"({pct(FX_FACTOR * exp_b, total_assets)}).")
    elif fx_excess:
        r.fx_requirement_after_exclusion = r.fx_requirement
        r.add("   The FX exposure comes from negative (short) positions, so the excess is not caused by assets; "
              "no exclusion applied. Check with an actuary.")
    else:
        r.fx_requirement_after_exclusion = r.fx_requirement
        r.add("   No FX concentration excess, so nothing is excluded; the requirement is unchanged.")
    if r.concentration_adjustment - fx_adjustment > 0:
        r.add("   The other excess assets must also be excluded from their own C2 risk requirements (equity, "
              "property, counterparty...), which this calculator does not compute.")

    after = ("after para 4.1.8" if r.fx_requirement_after_exclusion != r.fx_requirement
             else "(no para 4.1.8 recalculation)" if fx_excess else "")
    r.add("")
    r.add(f"Summary: FX mismatch risk requirement {m(r.fx_requirement_after_exclusion)} "
          f"({pct(r.fx_requirement_after_exclusion, total_assets)} of total assets) {after}; ".replace(" ;", ";")
          + f"adjustment for asset concentration {m(r.concentration_adjustment)} "
          f"({pct(r.concentration_adjustment, total_assets)}), deducted from financial resources.")
    if method == "total_less_limit" and fx_excess:
        r.add("FX adjustment method: total asset value less the 40% limit (set by the user, FX_ADJUSTMENT_METHOD "
              "in calculator.py). Have an actuary confirm it, and how para 4.1.8 applies, before relying on it.")
    else:
        r.add("The para 4.1.8 step is an interpretation of the notice; have an actuary confirm before relying on it.")
    return r


# =====================================================================================================
# C2 investment risk requirement (MAS 133 paras 4.3.1-4.3.7). Amounts in SGD millions, rates as decimals.
# Each module returns (requirement, steps) so callers can show the working. Interpretation choices are
# marked "Assumption:" in the steps; an actuary should confirm them.
# =====================================================================================================

# ---- Equity (para 4.3.2.1, Tables 4D-I / 4D-II, Guideline 4.12C) ----
EQUITY_STRESS = {"A": 0.35, "B": 0.50, "A_infra": 0.35, "B_infra": 0.40}   # A' / B' = qualifying infrastructure
EQUITY_SEGMENT_CORR = {"A": 0.90, "B": 0.75}                                # A with A', B with B'
CCA_BANDS = (0.05, 0.0, -0.05, -0.10)                                       # Table 4D-IV: allowed CCA values


def equity_requirement(holdings: list[dict], cca_a: float = 0.0, cca_b: float = 0.0) -> tuple[float, list[str]]:
    """holdings: [{"segment": "A" | "B" | "A_infra" | "B_infra", "amount": SGD m, "name": optional}].
    Segment A = listed in Developed Markets (MSCI World constituents); B = other markets, unlisted, private
    equity, hedge funds, commodities. cca_a / cca_b: the Table 4D-IV cyclical adjustment added to the A / B
    stress (not to the infrastructure stresses)."""
    for c in (cca_a, cca_b):
        if c not in CCA_BANDS:
            raise ValueError(f"CCA must be one of {CCA_BANDS}")
    stress = dict(EQUITY_STRESS, A=EQUITY_STRESS["A"] + cca_a, B=EQUITY_STRESS["B"] + cca_b)
    seg = {k: 0.0 for k in EQUITY_STRESS}
    steps = []
    for h in holdings:
        s = h["segment"]
        if s not in stress:
            raise ValueError(f"segment must be one of {list(stress)}")
        charge = h["amount"] * stress[s]
        seg[s] += charge
        steps.append(f"   {h.get('name', s)}: {m(h['amount'])} x {stress[s]:.0%} (segment {s}) = {m(charge)}")
    dm = math.sqrt(seg["A"] ** 2 + seg["A_infra"] ** 2 + 2 * 0.90 * seg["A"] * seg["A_infra"])
    other = math.sqrt(seg["B"] ** 2 + seg["B_infra"] ** 2 + 2 * 0.75 * seg["B"] * seg["B_infra"])
    steps.append(f"   Developed markets: (A {m(seg['A'])}, A' {m(seg['A_infra'])}, corr 0.9) = {m(dm)}")
    steps.append(f"   Other markets: (B {m(seg['B'])}, B' {m(seg['B_infra'])}, corr 0.75) = {m(other)}")
    steps.append(f"   Equity investment risk requirement = {m(dm)} + {m(other)} = {m(dm + other)} (para 4.3.2.1 a)iii))")
    return dm + other, steps


def cis_no_look_through(amount: float) -> float:
    """Appendix 4B para 5: a CIS not looked through takes a 50% charge on market value."""
    return 0.50 * amount


# ---- Interest rate mismatch (para 4.3.3.2, Table 4E) ----
# term in years -> (upward %, downward %) of the base yield
IR_FACTORS = {0.25: (1.00, -0.75), 0.5: (1.00, -0.70), 1: (1.00, -0.70), 2: (1.00, -0.70), 3: (0.95, -0.65),
              4: (0.95, -0.65), 5: (0.90, -0.60), 6: (0.85, -0.55), 7: (0.80, -0.50), 8: (0.80, -0.50),
              9: (0.75, -0.45), 10: (0.70, -0.40), 11: (0.65, -0.40), 12: (0.60, -0.35), 13: (0.60, -0.35),
              14: (0.55, -0.30), 15: (0.50, -0.30), 16: (0.45, -0.30), 17: (0.40, -0.30), 18: (0.35, -0.25),
              19: (0.30, -0.25), 20: (0.25, -0.25)}
IR_CAP = 0.02   # para 4.3.3.2 b)(i)C: absolute adjustment at most 200bp


def ir_factors(term: float) -> tuple[float, float]:
    """Closest Table 4E term (para 4.3.3.2 b)(iv)); 20 years and above use the 20-year row. Where a term is
    exactly halfway between two rows, the row with the larger shocks is used (Assumption: the notice does
    not say)."""
    if term >= 20:
        return IR_FACTORS[20]
    best = min(abs(k - term) for k in IR_FACTORS)
    rows = [IR_FACTORS[k] for k in IR_FACTORS if abs(abs(k - term) - best) < 1e-9]
    return max(rows, key=lambda r: abs(r[0]) + abs(r[1]))


def ir_shocks(term: float, base_yield: float) -> tuple[float, float]:
    """Absolute (upward, downward) yield moves for a cash flow at `term`, capped at 200bp each way."""
    up, down = ir_factors(term)
    return min(up * base_yield, IR_CAP), min(abs(down) * base_yield, IR_CAP)


def interest_rate_requirement(assets: list[dict], liabilities: list[dict]) -> tuple[float, str, list[str]]:
    """Simplified (duration) approach of para 4.3.3.2 c)(i).
    assets / liabilities: [{"value": SGD m, "duration": years, "yield": base yield, "term": optional, "name"}].
    `term` picks the Table 4E row (fixed-rate bonds: term to final maturity, para 4.3.3.2 b)(ivA)); it
    defaults to the duration, which is an approximation. Returns (requirement, dominant scenario, steps).
    The change in value is -value x duration x shift, with no convexity."""
    steps, d_up, d_dn = [], 0.0, 0.0
    for side, items, sign in (("Asset", assets, 1), ("Liability", liabilities, -1)):
        for x in items:
            term = x.get("term", x["duration"])
            up, dn = ir_shocks(term, x["yield"])
            ch_up, ch_dn = -x["value"] * x["duration"] * up, x["value"] * x["duration"] * dn
            d_up += sign * ch_up
            d_dn += sign * ch_dn
            steps.append(f"   {side} {x.get('name', '')} {m(x['value'])}, duration {x['duration']}, term {term}y, "
                         f"yield {x['yield']:.2%}: up {up * 1e4:.0f}bp -> {m(ch_up)}, down {dn * 1e4:.0f}bp -> "
                         f"{m(ch_dn)}".replace("  ", " "))
            if "term" not in x:
                steps.append("      Assumption: Table 4E term taken as the duration (the notice uses the cash flow "
                             "term, e.g. term to final maturity).")
    req = max(0.0, -d_up, -d_dn)
    scenario = "down" if -d_dn >= -d_up else "up"
    steps.append(f"   Change in net assets: up {m(d_up)}, down {m(d_dn)}")
    steps.append(f"   Interest rate mismatch risk requirement = larger fall = {m(req)} ({scenario} scenario, "
                 "para 4.3.3.2 b)(iii))")
    return req, scenario, steps


# ---- Interest rate mismatch, discounted cash flow approach (para 4.3.3.2 b), Guideline 4.14) ----
# Each cash flow is discounted at base spot rate + unshocked spread, annual compounding. The Table 4E shock for
# the cash flow's own (closest) term is applied to the base curve only: the government curve for assets, the
# risk-free spot curve for liabilities (para 4.3.3.2 b)(i)A). Spreads (an asset's credit spread, a liability's
# MA or IP spread) are not shocked.

def curve_rate(curve, t: float) -> float:
    """curve: a flat rate, or {term: spot rate} with linear interpolation and flat extrapolation."""
    if isinstance(curve, (int, float)):
        return float(curve)
    pts = sorted(curve.items())
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, r0), (t1, r1) in zip(pts, pts[1:]):
        if t <= t1:
            return r0 + (r1 - r0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


def present_value(cash_flows: list[tuple[float, float]], curve, spread: float = 0.0, shock: str | None = None,
                  extra: float = 0.0) -> float:
    """shock: None, "up" or "down" (Table 4E per cash flow, capped at 200bp); extra: a parallel add-on (credit
    spread shock)."""
    total = 0.0
    for t, amount in cash_flows:
        base = curve_rate(curve, t)
        if shock:
            up, dn = ir_shocks(t, base)
            base = base + up if shock == "up" else base - dn
        total += amount / (1 + base + spread + extra) ** t
    return total


def bond_cash_flows(face: float, coupon: float, maturity: int) -> list[tuple[float, float]]:
    """Annual-coupon fixed-rate bullet bond."""
    return [(t, face * coupon + (face if t == maturity else 0.0)) for t in range(1, maturity + 1)]


def duration(cash_flows: list[tuple[float, float]], curve, spread: float = 0.0) -> float:
    """Modified duration by a 1bp parallel bump (for reporting and calibration)."""
    p0 = present_value(cash_flows, curve, spread)
    return (p0 - present_value(cash_flows, curve, spread, extra=1e-4)) / p0 / 1e-4


def scale_to(cash_flows: list[tuple[float, float]], value: float, curve, spread: float = 0.0):
    """Scale cash flows so their base present value equals `value` (e.g. a fund's market value)."""
    k = value / present_value(cash_flows, curve, spread)
    return [(t, a * k) for t, a in cash_flows]


def dcf_interest_rate_requirement(assets: list[dict], liabilities: list[dict]) -> tuple[float, str, list[str]]:
    """assets / liabilities: [{"cash_flows": [(years, amount)], "curve": flat rate or {term: rate},
    "spread": unshocked add-on, "name"}]. Returns (requirement, dominant scenario, steps)."""
    steps, d_up, d_dn = [], 0.0, 0.0
    for side, items, sign in (("Asset", assets, 1), ("Liability", liabilities, -1)):
        for x in items:
            cf, cv, sp = x["cash_flows"], x["curve"], x.get("spread", 0.0)
            base, up, dn = (present_value(cf, cv, sp, s) for s in (None, "up", "down"))
            d_up += sign * (up - base)
            d_dn += sign * (dn - base)
            steps.append(f"   {side} {x.get('name', '')}: value {m(base)} (modified duration "
                         f"{duration(cf, cv, sp):.1f}); up -> {m(up)} ({m(up - base)}), down -> {m(dn)} "
                         f"({m(dn - base)})".replace("  ", " "))
    req = max(0.0, -d_up, -d_dn)
    scenario = "down" if -d_dn >= -d_up else "up"
    steps.append(f"   Change in net assets: up {m(d_up)}, down {m(d_dn)}")
    steps.append(f"   Interest rate mismatch risk requirement = larger fall = {m(req)} ({scenario} scenario, "
                 "para 4.3.3.2 b)(iii)), discounted cash flow approach")
    return req, scenario, steps


# ---- Credit spread (para 4.3.3.3, Tables 4F-II / 4F-III, Appendix 4K) ----
# long-term credit quality class -> bp shock for (term <= 5y, 5y < term <= 10y, term > 10y)
SPREAD_LONG = {"A": (105, 95, 90), "B": (120, 115, 95), "C": (165, 145, 125), "D": (245, 230, 215),
               "E": (405, 365, 355), "F": (540, 500, 475), "unrated": (325, 298, 285)}
RATING_CLASS = {"AAA": "A", "AA": "B", "A": "C", "BBB": "D", "BB": "E", "B": "F", "CCC": "F"}  # Appendix 4K


def spread_shock(credit_class: str, term: float) -> float:
    """Shock in bp. credit_class: "A".."G" (Appendix 4K class, G uses "F and below"), "unrated",
    "statutory_board" (50% of class A, para 4.3.3.3 e)) or "sovereign_exempt" (central government or central
    bank rated class C or better, para 4.3.3.3 i): no charge)."""
    if credit_class == "sovereign_exempt":
        return 0.0
    col = 0 if term <= 5 else 1 if term <= 10 else 2
    if credit_class == "statutory_board":
        return 0.5 * SPREAD_LONG["A"][col]
    return SPREAD_LONG["F" if credit_class == "G" else credit_class][col]


def credit_spread_requirement(holdings: list[dict]) -> tuple[float, list[str]]:
    """holdings: [{"value", "duration", "class", "term": optional (defaults to duration), "name"}].
    Fall in value = value x spread duration x shock (duration approach; spread duration taken as duration)."""
    total, steps = 0.0, []
    for h in holdings:
        term = h.get("term", h["duration"])
        bp = spread_shock(h["class"], term)
        loss = h["value"] * h["duration"] * bp / 1e4
        total += loss
        steps.append(f"   {h.get('name', h['class'])}: {m(h['value'])} x {h['duration']} x {bp:g}bp "
                     f"(class {h['class']}, term {term}y) = {m(loss)}")
    steps.append(f"   Credit spread risk requirement = {m(total)} (para 4.3.3.3 b)(iii))")
    return total, steps


def dcf_credit_spread_requirement(holdings: list[dict]) -> tuple[float, list[str]]:
    """holdings: [{"cash_flows", "curve" (government), "spread" (the bond's current credit spread), "class",
    "term": optional (defaults to the last cash flow, i.e. final maturity), "name"}]. Revalues each bond with the
    constant bp shock added to its yield curve (para 4.3.3.3 b)(ii))."""
    total, steps = 0.0, []
    for h in holdings:
        term = h.get("term", max(t for t, _ in h["cash_flows"]))
        bp = spread_shock(h["class"], term)
        base = present_value(h["cash_flows"], h["curve"], h.get("spread", 0.0))
        shocked = present_value(h["cash_flows"], h["curve"], h.get("spread", 0.0), extra=bp / 1e4)
        total += base - shocked
        steps.append(f"   {h.get('name', h['class'])}: value {m(base)}, +{bp:g}bp (class {h['class']}, term "
                     f"{term}y) -> {m(shocked)}; fall {m(base - shocked)}")
    steps.append(f"   Credit spread risk requirement = {m(total)} (para 4.3.3.3 b)(iii)), discounted cash flows")
    return total, steps


# ---- Property (para 4.3.4.1) ----
def property_requirement(direct: float = 0.0, cis_no_look_through_value: float = 0.0) -> float:
    return 0.30 * direct + 0.50 * cis_no_look_through_value


# ---- Counterparty default (para 4.3.6, Table 4H, Appendix 4H) ----
DEFAULT_CHARGE = {"A": 0.005, "B": 0.01, "C": 0.02, "D": 0.05, "E": 0.105, "F": 0.20, "G": 0.485,
                  "unrated": 0.0775}
FX_CREDIT_EXPOSURE_FACTOR = ((14 / 365, 0.0), (1, 0.01), (5, 0.05), (float("inf"), 0.075))  # Appendix 4H (3)


def fx_forward_exposure(notional: float, maturity_years: float, replacement_cost: float = 0.0,
                        original_maturity_years: float | None = None) -> float:
    """Credit equivalent amount of an FX contract (para 4.3.6.4 d)): replacement cost if positive plus potential
    credit exposure (notional x Appendix 4H factor). Original maturity of 14 days or less: zero."""
    if (original_maturity_years if original_maturity_years is not None else maturity_years) <= 14 / 365:
        return 0.0
    factor = next(f for limit, f in FX_CREDIT_EXPOSURE_FACTOR[1:] if maturity_years <= limit)
    return max(replacement_cost, 0.0) + notional * factor


def default_requirement(exposures: list[dict]) -> tuple[float, list[str]]:
    """exposures: [{"amount": credit-equivalent SGD m, "class": Appendix 4L class or "unrated", "name"}]."""
    total, steps = 0.0, []
    for e in exposures:
        r = e["amount"] * DEFAULT_CHARGE[e["class"]]
        total += r
        steps.append(f"   {e.get('name', '')} {m(e['amount'])} x {DEFAULT_CHARGE[e['class']]:.2%} "
                     f"(class {e['class']}) = {m(r)}".replace("  ", " "))
    steps.append(f"   Counterparty default risk requirement = {m(total)}")
    return total, steps


# ---- Miscellaneous (para 4.3.7.1) ----
def misc_requirement(other_assets: float) -> float:
    """8% of assets with no other C2 charge (excluding cash and non-standard instruments)."""
    return 0.08 * other_assets


# ---- Aggregation (paras 4.3.1.1, 4.3.1.4, Tables 4C-I / 4C-II) ----
MARKET_MODULES = ("equity", "interest_rate", "credit_spread", "property", "fx")
CORR_UP = ((1, .1, .8, .8, .1), (.1, 1, .1, .1, .1), (.8, .1, 1, .5, .1), (.8, .1, .5, 1, .1), (.1, .1, .1, .1, 1))
CORR_DOWN = ((1, .5, .8, .8, .1), (.5, 1, .5, .25, .1), (.8, .5, 1, .5, .1), (.8, .25, .5, 1, .1),
             (.1, .1, .1, .1, 1))
CORR_MARKET_DEFAULT = 0.5


@dataclass
class C2Result:
    market: float
    c2: float
    undiversified: float
    contributions: dict[str, float]     # Euler allocation of c2; sums to c2
    lines: list[str] = field(default_factory=list)


def aggregate_c2(modules: dict[str, float], scenario: str, default: float = 0.0, misc: float = 0.0) -> C2Result:
    """modules: requirement per MARKET_MODULES key (missing = 0). scenario: "up" or "down", the interest rate
    scenario that gave the interest rate requirement, which picks Table 4C-I or 4C-II."""
    if scenario not in ("up", "down"):
        raise ValueError("scenario must be 'up' or 'down'")
    unknown = set(modules) - set(MARKET_MODULES)
    if unknown:
        raise ValueError(f"unknown modules {unknown}; expected {MARKET_MODULES}")
    corr = CORR_DOWN if scenario == "down" else CORR_UP
    v = [modules.get(k, 0.0) for k in MARKET_MODULES]
    market = math.sqrt(sum(corr[i][j] * v[i] * v[j] for i in range(5) for j in range(5)))
    rho = CORR_MARKET_DEFAULT
    core = math.sqrt(market ** 2 + default ** 2 + 2 * rho * market * default)
    c2 = misc + core
    # Euler allocation: each term's marginal share of the square-root parts; misc adds 1:1.
    contrib = {}
    if core:
        for i, k in enumerate(MARKET_MODULES):
            dm = sum(corr[i][j] * v[j] for j in range(5)) / market if market else 0.0
            contrib[k] = v[i] * dm * (market + rho * default) / core
        contrib["default"] = default * (default + rho * market) / core
    else:
        contrib = {k: 0.0 for k in (*MARKET_MODULES, "default")}
    contrib["misc"] = misc
    undiv = sum(v) + default + misc
    table = "4C-II (downward)" if scenario == "down" else "4C-I (upward)"
    lines = [f"   Market modules: " + ", ".join(f"{k} {m(x)}" for k, x in zip(MARKET_MODULES, v)),
             f"   C2 market = sqrt(sum corr x Mi x Mj), Table {table} = {m(market)} (para 4.3.1.4)",
             f"   C2 = misc {m(misc)} + sqrt({m(market)}^2 + default {m(default)}^2 + 2 x 0.5 x market x default)"
             f" = {m(c2)} (para 4.3.1.1)",
             f"   Undiversified sum {m(undiv)}; diversification benefit {m(undiv - c2)}"]
    return C2Result(market, c2, undiv, contrib, lines)
