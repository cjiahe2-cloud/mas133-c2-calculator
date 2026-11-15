"""Deterministic MAS 133 calculations the chat model kept getting wrong when it chained the rules itself:

- foreign currency mismatch risk requirement (paras 4.3.5.3-4.3.5.7)
- adjustment for asset concentration (Appendix 5E paras 16-17B, Table 2-I), deducted from financial
  resources (para 5.1(a)(ii))
- the FX requirement recalculated after excluding the assets that cause the concentration adjustment
  from the C2 calculation (para 4.1.8)

All amounts are SGD millions. How the FX part of the adjustment is worked out is set by FX_ADJUSTMENT_METHOD
below (the user's reading: total asset value less the 40% limit). The interpretation of para 4.1.8 (which assets are excluded, and whether the
10% / 20% deduction base shrinks with them) is our reading of the text: it should be confirmed by an actuary.
"""
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
