"""Calculator check (no OpenAI calls, free): uv run python calc_test.py

Part 1 (FX + concentration): expected figures are worked by hand from MAS 133 paras 4.3.5.3-4.3.5.7, 4.1.8 and
Appendix 5E paras 16-17B.
Cases without a method use FX_ADJUSTMENT_METHOD's default ("total_less_limit": the FX adjustment is total asset
value less the 40% limit, once FX exposure is over the limit).
"""
from insurance_mas_rag import calculator as calc
from insurance_mas_rag.calculator import calculate

SG = dict(fund_type="singapore", total_assets=100)
CASES = [
    # (description, kwargs, FX requirement before 4.1.8, after 4.1.8, concentration adjustment)
    # ---- default method: total asset value less the limit ----
    ("User's example: 100m, 60% USD 40% SGD -> 100m - 40m", dict(SG, fx_positions={"USD": 60, "SGD": 40}),
     6.0, 6.0, 60),
    ("Singapore fund, 20/30/50 SGD/EUR/USD", dict(SG, fx_positions={"SGD": 20, "EUR": 30, "USD": 50}),
     8.4, 8.4, 60),
    ("Singapore fund, 24/26/50", dict(SG, fx_positions={"SGD": 24, "EUR": 26, "USD": 50}), 7.92, 7.92, 60),
    ("Offshore fund, 24/26/50: 20% deduction, no FX limit", dict(fund_type="offshore", total_assets=100,
                                                                 fx_positions={"EUR": 26, "USD": 50}),
     6.72, 6.72, 0),
    ("Exposure within the 40% limit (45m - 10m = 35m): no adjustment", dict(SG, fx_positions={"USD": 45}),
     4.2, 4.2, 0),
    ("Exposure below the 10% deduction", dict(SG, fx_positions={"USD": 8}), 0, 0, 0),
    ("Short positions: higher of long 20 and short 30", dict(SG, fx_positions={"USD": 20, "EUR": -30}),
     2.4, 2.4, 0),
    ("Small fund: para 17B $5m minimum limit -> 10m - 5m", dict(fund_type="singapore", total_assets=10,
                                                                fx_positions={"USD": 7}), 0.72, 0.72, 5),
    ("FX 80m plus property 40m over the 35% limit (60m + 5m)", dict(SG, fx_positions={"USD": 80},
                                                                    exposures=[{"item": "property", "amount": 40}]),
     8.4, 8.4, 65),
    # ---- alternative method: only the excess over the limit, then para 4.1.8 ----
    ("[excess] User's example: 50m exposure - 40m", dict(SG, fx_positions={"USD": 60}, method="excess"),
     6.0, 4.8, 10),
    ("[excess] Singapore fund, 20/30/50", dict(SG, fx_positions={"EUR": 30, "USD": 50}, method="excess"),
     8.4, 4.8, 30),
    ("[excess] Singapore fund, 24/26/50", dict(SG, fx_positions={"EUR": 26, "USD": 50}, method="excess"),
     7.92, 4.8, 26),
    ("[excess] Small fund, $5m minimum", dict(fund_type="singapore", total_assets=10, fx_positions={"USD": 7},
                                              method="excess"), 0.72, 0.6, 1),
]

failed = 0
for name, kwargs, before, after, adj in CASES:
    r = calculate(**kwargs)
    got = (round(r.fx_requirement, 4), round(r.fx_requirement_after_exclusion, 4),
           round(r.concentration_adjustment, 4))
    ok = got == (before, after, adj)
    failed += not ok
    print(f"{'✓' if ok else '✗'} {name}: requirement {got[0]}m -> {got[1]}m after 4.1.8, adjustment {got[2]}m"
          + ("" if ok else f"  (expected {before}, {after}, {adj})"))
total = len(CASES)
print(f"\nPart 1: {len(CASES) - failed} of {len(CASES)} passed.\n")

# =====================================================================================================
# Part 2: C2 modules. "MAS" = figure published in the notice itself; "hand" = worked by hand from the
# notice's tables (the working is in the comment).
# =====================================================================================================


def check(name: str, got: float, expected: float, places: int = 2) -> None:
    global failed, total
    total += 1
    ok = round(got, places) == round(expected, places)
    failed += not ok
    print(f"{'✓' if ok else '✗'} {name}: {round(got, places)}" + ("" if ok else f"  (expected {expected})"))


print("-- Equity (para 4.3.2.1)")
# MAS Guideline 4.12C: SGX 300 (A), S&P 500 qualifying infra 200 (A'), SSE 300 (B), Brazil + Indonesia
# qualifying infra 100 + 100 (B'). MAS: DM 170.7, other 216.6, total 387.3.
eq, _ = calc.equity_requirement([{"segment": "A", "amount": 300}, {"segment": "A_infra", "amount": 200},
                                 {"segment": "B", "amount": 300}, {"segment": "B_infra", "amount": 200}])
check("MAS Guideline 4.12C worked example (MAS: 387.3)", eq, 387.3, 1)
# MAS Appendix 4B Guideline 2: CIS mandate 20-30% Singapore listed / 70-80% other -> 20/80 -> 47%.
eq, _ = calc.equity_requirement([{"segment": "A", "amount": 20}, {"segment": "B", "amount": 80}])
check("MAS Appendix 4B Guideline 2 CIS mandate (MAS: 47%)", eq, 47)
check("Appendix 4B para 5: CIS not looked through, 100 x 50%", calc.cis_no_look_through(100), 50)
# hand: 400 x 35% + (100 + 50) x 50% = 140 + 75
eq, _ = calc.equity_requirement([{"segment": "A", "amount": 400}, {"segment": "B", "amount": 100},
                                 {"segment": "B", "amount": 50}])
check("Model portfolio v1 (hand: 140 + 75)", eq, 215)
# hand: CCA +5% on segment A -> 40%
eq, _ = calc.equity_requirement([{"segment": "A", "amount": 100}], cca_a=0.05)
check("CCA +5% on Developed Markets (hand: 100 x 40%)", eq, 40)

print("-- Interest rate mismatch (para 4.3.3.2, Table 4E)")
check("8y at 3%: up 80% = 240bp, capped at 200bp", calc.ir_shocks(8, 0.03)[0], 0.02, 6)
check("8y at 3%: down 50% = 150bp", calc.ir_shocks(8, 0.03)[1], 0.015, 6)
check("20y at 3%: up 25% = 75bp", calc.ir_shocks(20, 0.03)[0], 0.0075, 6)
check("30y uses the 20y row: down 25% = 75bp", calc.ir_shocks(30, 0.03)[1], 0.0075, 6)
check("1 month uses the 3m row: down 75% x 3% = 225bp, capped at 200bp", calc.ir_shocks(1 / 12, 0.03)[1], 0.02, 6)
check("10.4y uses the 10y row: up 70% x 2% = 140bp", calc.ir_shocks(10.4, 0.02)[0], 0.014, 6)
check("2.5y tie between 2y (70%) and 3y (65%): larger down shock", calc.ir_factors(2.5)[1], -0.70, 6)
A1 = [{"value": 450, "duration": 8, "yield": 0.03}]
L1 = [{"value": 900, "duration": 20, "yield": 0.03}]
# hand: up -450x8x2% + 900x20x0.75% = -72 + 135 = +63; down +450x8x1.5% - 135 = 54 - 135 = -81
ir, sc, _ = calc.interest_rate_requirement(A1, L1)
check("Model portfolio v1: down scenario loses 81 (hand)", ir, 81)
check("  ...and the down scenario is dominant", sc == "down", True)
ir, sc, _ = calc.interest_rate_requirement([{"value": 600, "duration": 8, "yield": 0.03}], L1)
check("Model portfolio v2: 72 - 135 = -63 (hand)", ir, 63)
# hand: assets 1000 D10 (70% x 3% = 210bp -> 200bp up; 40% -> 120bp down), liabilities 900 D5 (90% -> 200bp cap;
# 60% -> 180bp). Up: -200 + 90 = -110. Down: +120 - 81 = +39. Requirement 110, up scenario.
ir, sc, _ = calc.interest_rate_requirement([{"value": 1000, "duration": 10, "yield": 0.03}],
                                           [{"value": 900, "duration": 5, "yield": 0.03}])
check("Assets longer than liabilities: up scenario loses 110 (hand)", ir, 110)
check("  ...and the up scenario is dominant", sc == "up", True)
# hand: duration 8 but 12y to maturity -> 12y row: up 60% x 3% = 180bp -> 450x8x1.8% = 64.8 asset fall
ir, _, _ = calc.interest_rate_requirement([{"value": 450, "duration": 8, "term": 12, "yield": 0.03}], [])
check("Term to maturity (12y) sets the row, not duration (hand: 64.8)", ir, 64.8)

print("-- Interest rate mismatch, discounted cash flows (para 4.3.3.2 b))")
Z10 = [(10, 100)]
check("Curve interpolation: 2% at 1y, 4% at 10y -> 3% at 5.5y", calc.curve_rate({1: 0.02, 10: 0.04}, 5.5), 0.03, 6)
check("Zero-coupon 10y at 3%: 100 / 1.03^10", calc.present_value(Z10, 0.03), 100 / 1.03 ** 10, 6)
# up: 70% x 3% = 210bp -> capped 200bp -> 5%; down: 40% x 3% = 120bp -> 1.8%
ir, sc, _ = calc.dcf_interest_rate_requirement([{"cash_flows": Z10, "curve": 0.03}], [])
check("10y zero asset, up scenario: 100/1.03^10 - 100/1.05^10", ir, 100 / 1.03 ** 10 - 100 / 1.05 ** 10, 6)
# liability 20y zero: down 25% x 3% = 75bp -> 2.25%; liability rises, net assets fall
ir, sc, _ = calc.dcf_interest_rate_requirement([], [{"cash_flows": [(20, 100)], "curve": 0.03}])
check("20y zero liability, down scenario: 100/1.0225^20 - 100/1.03^20", ir, 100 / 1.0225 ** 20 - 100 / 1.03 ** 20, 6)
check("  ...and the down scenario is dominant", sc == "down", True)
# each cash flow takes its own row: 1y up 100% -> 300bp -> 200bp cap; 20y up 25% -> 75bp
ir, _, _ = calc.dcf_interest_rate_requirement([{"cash_flows": [(1, 100), (20, 100)], "curve": 0.03}], [])
check("Per-cash-flow rows: 1y +200bp (capped), 20y +75bp", ir,
      (100 / 1.03 + 100 / 1.03 ** 20) - (100 / 1.05 + 100 / 1.0375 ** 20), 6)
# credit spread 1% is discounted but not shocked: 3% x 70% = 210bp -> 200bp on the government curve only
ir, _, _ = calc.dcf_interest_rate_requirement([{"cash_flows": Z10, "curve": 0.03, "spread": 0.01}], [])
check("Spread not shocked: 100/1.04^10 - 100/1.06^10", ir, 100 / 1.04 ** 10 - 100 / 1.06 ** 10, 6)
check("Modified duration of a 10y zero at 3% (about 10 / 1.03)", calc.duration(Z10, 0.03), 9.7, 1)
check("Scale cash flows to a 900m value", calc.present_value(calc.scale_to([(20, 1)], 900, 0.03), 0.03), 900, 6)

print("-- Credit spread (para 4.3.3.3, Tables 4F-II / 4F-III)")
for cls, term, bp in (("C", 8, 145), ("E", 8, 365), ("unrated", 8, 298), ("A", 5, 105), ("A", 5.5, 95),
                      ("G", 12, 475), ("F", 3, 540), ("statutory_board", 3, 52.5), ("sovereign_exempt", 8, 0)):
    check(f"Shock class {cls}, {term}y = {bp}bp", calc.spread_shock(cls, term), bp)
cs, _ = calc.credit_spread_requirement([
    {"value": 100, "duration": 8, "class": "C"}, {"value": 100, "duration": 8, "class": "E"},
    {"value": 50, "duration": 8, "class": "unrated"}, {"value": 200, "duration": 8, "class": "sovereign_exempt"}])
check("Model portfolio v1 (hand: 11.6 + 29.2 + 11.92 + 0)", cs, 52.72)

# DCF: 10y zero, government 3% + spread 1%, class C (A rating), 5-10y bucket -> +145bp
cs, _ = calc.dcf_credit_spread_requirement([{"cash_flows": Z10, "curve": 0.03, "spread": 0.01, "class": "C"}])
check("DCF credit spread: 100/1.04^10 - 100/1.0545^10", cs, 100 / 1.04 ** 10 - 100 / 1.0545 ** 10, 6)
cs, _ = calc.dcf_credit_spread_requirement([{"cash_flows": calc.bond_cash_flows(100, 0.03, 12), "curve": 0.03,
                                             "class": "C"}])
check("DCF credit spread uses final maturity (12y -> >10y bucket, 125bp)", cs,
      calc.present_value(calc.bond_cash_flows(100, 0.03, 12), 0.03)
      - calc.present_value(calc.bond_cash_flows(100, 0.03, 12), 0.0425), 6)

print("-- Property, default, miscellaneous")
check("Property 100 direct x 30%", calc.property_requirement(direct=100), 30)
check("Property CIS not looked through 100 x 50%", calc.property_requirement(cis_no_look_through_value=100), 50)
check("FX forward 800m, 6 months: 800 x 1%", calc.fx_forward_exposure(800, 0.5), 8)
check("FX forward, original maturity 10 days: 0", calc.fx_forward_exposure(800, 10 / 365), 0)
check("FX forward 800m, 2 years: 800 x 5%", calc.fx_forward_exposure(800, 2), 40)
check("FX forward with +3m replacement cost: 3 + 8", calc.fx_forward_exposure(800, 0.5, replacement_cost=3), 11)
d, _ = calc.default_requirement([{"amount": 8, "class": "C"}])
check("Default: 8 x 2% (class C)", d, 0.16)
d, _ = calc.default_requirement([{"amount": 100, "class": "unrated"}])
check("Default: unrated 100 x 7.75%", d, 7.75)
check("Miscellaneous: 100 x 8%", calc.misc_requirement(100), 8)

print("-- Aggregation (paras 4.3.1.1, 4.3.1.4, Tables 4C-I / 4C-II)")
check("Equity alone: no diversification", calc.aggregate_c2({"equity": 100}, "down").c2, 100)
# hand: sqrt(30^2 + 40^2 + 2 x 0.1 x 30 x 40) = sqrt(2740)
check("Equity 30 + IR 40, upward (corr 0.1)", calc.aggregate_c2({"equity": 30, "interest_rate": 40}, "up").c2,
      52.345, 3)
# hand: sqrt(900 + 1600 + 2 x 0.5 x 1200) = sqrt(3700)
check("Equity 30 + IR 40, downward (corr 0.5)", calc.aggregate_c2({"equity": 30, "interest_rate": 40}, "down").c2,
      60.828, 3)
# hand: 10 + sqrt(60^2 + 80^2 + 2 x 0.5 x 60 x 80) = 10 + sqrt(14800)
check("Market 60 (equity) + default 80 + misc 10", calc.aggregate_c2({"equity": 60}, "down", 80, 10).c2, 131.655, 3)
v1 = calc.aggregate_c2({"equity": 215, "interest_rate": 81, "credit_spread": 52.72, "fx": 12}, "down", 0.16)
check("Model portfolio v1 C2", v1.c2, 310.51)
check("  ...contributions add up to C2", sum(v1.contributions.values()), v1.c2, 6)
v2 = calc.aggregate_c2({"equity": 155, "interest_rate": 63, "credit_spread": 58.52, "fx": 12}, "down", 0.16)
check("Model portfolio v2 C2", v2.c2, 245.42)

print(f"\n{total - failed} of {total} passed.")
print("\nNote: MAS Appendix 4B Guideline 4 (infrastructure CIS) adds 28% + 15% = 43% without the 0.75 correlation of "
      "para 4.3.2.1 a)iii), which would give 40.5%. The calculator follows para 4.3.2.1; ask MAS or an actuary which "
      "applies to a look-through CIS.")
raise SystemExit(1 if failed else 0)
