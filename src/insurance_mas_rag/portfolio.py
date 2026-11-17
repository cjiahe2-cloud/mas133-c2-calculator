"""Whole-portfolio MAS 133 C2 calculation for the public calculator page (calc_app.py).

Turns an asset allocation (% of the fund per asset class, bond durations), a simple liability
profile (value + duration) and an FX hedge ratio into the inputs of the calculator.py modules, then aggregates C2
and checks the Table 2-I concentration limits. Amounts in SGD millions, shares and rates as decimals.

Simplifications (shown on the page): one base yield for every asset and liability; the Table 4E row is taken from
the duration; the currency mix is entered directly (SGD % of the fund, of which FX forwards %; the rest is USD), not
taken from the asset classes (the user's choice); liabilities are SGD;
every single-issuer exposure is assumed to be within its Table 2-I limit (the user's choice), so only the aggregate
limits (unlisted equity, property, FX) and the FX forward bank are checked.
"""
from dataclasses import dataclass, field

from insurance_mas_rag.calculator import (LIMITS, C2Result, aggregate_c2, calculate, credit_spread_requirement,
                                          default_requirement, equity_requirement, fx_forward_exposure,
                                          interest_rate_requirement, limit_amount, property_requirement, FX_LIMIT,
                                          RATING_CLASS)

# kind: bond / equity / property. credit: Appendix 4K class for Table 4F (sovereign_exempt = para 4.3.3.3 i)).
# ratings: the ratings a user can pick for the class (mapped to the Appendix 4K class by RATING_CLASS).
# segment: Table 4D-I equity segment. unlisted: counts towards the Table 2-I (2)(c) aggregate unlisted equity limit.
ASSET_CLASSES = {
    "sgs": dict(label="Singapore government bonds", kind="bond", credit="sovereign_exempt"),
    "ust": dict(label="Developed-market government bonds", kind="bond", credit="sovereign_exempt"),
    "corp_ig": dict(label="Investment-grade corporate bonds", kind="bond",
                    ratings=("AAA", "AA", "A", "BBB")),
    "corp_hy": dict(label="High-yield corporate bonds", kind="bond", ratings=("BB", "B", "CCC")),
    "unrated": dict(label="Unrated bonds", kind="bond", credit="unrated"),
    "eq_dm": dict(label="Developed-market listed equity", kind="equity", segment="A"),
    "eq_em": dict(label="Emerging-market listed equity", kind="equity", segment="B"),
    "pe": dict(label="Private equity", kind="equity", segment="B", unlisted=True),
    "property": dict(label="Property (Singapore)", kind="property"),
}

# Rating of the FX forward counterparty -> Appendix 4L counterparty risk class for the Table 4H default charge
# (Appendix 4L uses the same rating bands as Appendix 4K, checked against the notice). AAA to B is the user's
# range: insurers rarely trade derivatives with CCC or unrated banks.
FORWARD_RATING = {r: RATING_CLASS[r] for r in ("AAA", "AA", "A", "BBB", "BB", "B")}

# The user's sample insurer ("Model portfolio v1"): SGD 1,000m Singapore-policies fund, 80% SGD (10% natural SGD +
# 70% of the fund hedged back to SGD with FX forwards selling USD for SGD), 20% USD. No concentration limit is breached, and every
# single-issuer exposure is assumed to be within its limit.
SAMPLE_PORTFOLIO = {
    "total_assets": 1000.0,
    "base_yield": 0.03,
    "allocation": {"sgs": .10, "ust": .10, "corp_ig": .10, "corp_hy": .10, "unrated": .05,
                   "eq_dm": .40, "eq_em": .10, "pe": .05, "property": 0.0},
    "duration": {"sgs": 8.0, "ust": 8.0, "corp_ig": 8.0, "corp_hy": 8.0, "unrated": 8.0},
    "rating": {"corp_ig": "A", "corp_hy": "BB"},
    "liabilities": {"value": 900.0, "duration": 20.0},
    "sgd_share": .80,             # SGD exposure after hedging, as a share of the fund; the rest is USD
    "hedged": .70,                # FX forward notional as a share of the fund (part of the SGD share)
    "forward_months": 6,
    "forward_rating": "A",
}


@dataclass
class LimitRow:
    item: str            # Table 2-I item, e.g. "(2)(a)"
    covers: str
    holding: str         # which input it checks
    exposure: float
    limit: float
    adjustment: float

    @property
    def breach(self) -> bool:
        return self.exposure > self.limit + 1e-9


@dataclass
class Analysis:
    modules: dict[str, float]          # standalone requirement per C2 module
    steps: dict[str, list[str]]        # working per module
    scenario: str                      # dominant interest rate scenario
    c2: C2Result
    limits: list[LimitRow]
    concentration_adjustment: float
    usd_share: float                   # non-SGD (USD) share of the fund after hedging
    fx_lines: list[str] = field(default_factory=list)


def weighted_durations(allocation: dict[str, float], duration: dict[str, float]) -> tuple[float | None, float]:
    """Market-value weighted effective duration of (the bonds, all assets). Equity and property count as zero in
    the portfolio figure. The bond figure is None when there are no bonds."""
    bonds = {k: w for k, w in allocation.items() if ASSET_CLASSES[k]["kind"] == "bond"}
    dollar = sum(w * duration.get(k, 0.0) for k, w in bonds.items())
    total = sum(allocation.values())
    return (dollar / sum(bonds.values()) if sum(bonds.values()) else None), (dollar / total if total else 0.0)


def credit_class(key: str, p: dict) -> str:
    """Appendix 4K class of an asset class: fixed, or from the rating the user picked."""
    c = ASSET_CLASSES[key]
    return RATING_CLASS[p["rating"][key]] if "ratings" in c else c["credit"]


def validate(p: dict) -> list[str]:
    errors = []
    for k, c in ASSET_CLASSES.items():
        if "ratings" in c and p["rating"].get(k) not in c["ratings"]:
            errors.append(f"{c['label']}: rating must be one of {', '.join(c['ratings'])}.")
    alloc = p["allocation"]
    if abs(sum(alloc.values()) - 1) > 1e-6:
        errors.append(f"Allocations add up to {sum(alloc.values()):.1%}; they must add up to 100%.")
    if any(v < 0 for v in alloc.values()):
        errors.append("Allocations can't be negative.")
    if not 0 <= p["sgd_share"] <= 1:
        errors.append("The SGD share must be between 0% and 100%.")
    if p["hedged"] > p["sgd_share"] + 1e-9:
        errors.append(f"FX forwards ({p['hedged']:.0%}) can't be more than the SGD share ({p['sgd_share']:.0%}): "
                      "they are part of it.")
    if p["total_assets"] <= 0:
        errors.append("Total assets must be positive.")
    return errors


def analyse(p: dict) -> Analysis:
    ta, y = p["total_assets"], p["base_yield"]
    amt = {k: p["allocation"].get(k, 0.0) * ta for k in ASSET_CLASSES}
    bonds = [k for k, c in ASSET_CLASSES.items() if c["kind"] == "bond" and amt[k] > 0]
    label = {k: c["label"] for k, c in ASSET_CLASSES.items()}
    modules, steps = {}, {}

    # Equity (para 4.3.2.1)
    holdings = [{"segment": c["segment"], "amount": amt[k], "name": c["label"]}
                for k, c in ASSET_CLASSES.items() if c["kind"] == "equity" and amt[k] > 0]
    modules["equity"], steps["equity"] = equity_requirement(holdings)

    # Interest rate mismatch, duration approach (para 4.3.3.2 c)(i))
    assets = [{"value": amt[k], "duration": p["duration"][k], "yield": y, "name": label[k]} for k in bonds]
    liab = p["liabilities"]
    liabilities = ([{"value": liab["value"], "duration": liab["duration"], "yield": y, "name": "policy liabilities"}]
                   if liab["value"] > 0 else [])
    modules["interest_rate"], scenario, steps["interest_rate"] = interest_rate_requirement(assets, liabilities)

    # Credit spread (para 4.3.3.3)
    modules["credit_spread"], steps["credit_spread"] = credit_spread_requirement(
        [{"value": amt[k], "duration": p["duration"][k], "class": credit_class(k, p), "name": label[k]}
         for k in bonds])

    # Property (para 4.3.4.1)
    modules["property"] = property_requirement(direct=amt["property"])
    steps["property"] = [f"   Direct property {amt['property']:,.2f}m x 30% = {modules['property']:,.2f}m"]

    # FX mismatch (para 4.3.5) and Table 2-I concentration (Appendix 5E) via calculate()
    usd_share = 1 - p["sgd_share"]
    notional = p["hedged"] * ta
    usd_open = usd_share * ta              # net open USD position after the forwards
    rows = []   # (Table 2-I item, exposure, what it checks); single-issuer limits are assumed met
    unlisted = sum(amt[k] for k, c in ASSET_CLASSES.items() if c.get("unlisted"))
    if unlisted:
        rows.append(("equity_unlisted_aggregate", unlisted, "All private equity"))
    if amt["property"]:
        rows.append(("property", amt["property"], "All property"))
    credit_eq = fx_forward_exposure(notional, p["forward_months"] / 12) if notional else 0.0
    if credit_eq:
        rows.append(("counterparty_approved_fi", credit_eq, "FX forward bank (credit equivalent amount)"))
    exposures = [{"item": item, "amount": exposure} for item, exposure, _ in rows]
    r = calculate("singapore", ta, {"USD": usd_open}, exposures=exposures)
    modules["fx"] = r.fx_requirement_after_exclusion
    blank = r.lines.index("")
    steps["fx"] = r.lines[1:blank]

    # Counterparty default (para 4.3.6): the FX forwards
    cls = FORWARD_RATING[p["forward_rating"]]
    default, steps["default"] = default_requirement(
        [{"amount": credit_eq, "class": cls, "name": f"FX forwards {notional:,.0f}m notional, "
                                                     f"{p['forward_months']} months, {p['forward_rating']}-rated bank"}]
        if credit_eq else [])
    steps["default"].insert(0, f"   Credit equivalent amount = notional x Appendix 4H factor = {credit_eq:,.2f}m")

    c2 = aggregate_c2(modules, scenario, default=default)
    modules["default"] = default

    limits = []
    fx_lim, _ = limit_amount(FX_LIMIT, ta)
    other_adj = 0.0
    for item, exposure, holding in rows:
        rate, code, covers = LIMITS[item]
        lim, _ = limit_amount(rate, ta)
        adj = max(0.0, exposure - lim)
        other_adj += adj
        limits.append(LimitRow(code, covers, holding, exposure, lim, adj))
    limits.append(LimitRow("(5)", "foreign currency risk exposure (para 4.3.5.6)", "Unhedged USD less the 10% "
                           "deduction", r.fx_exposure, fx_lim, r.concentration_adjustment - other_adj))
    limits.sort(key=lambda row: row.item)
    return Analysis(modules, steps, scenario, c2, limits, r.concentration_adjustment, usd_share, r.lines)
