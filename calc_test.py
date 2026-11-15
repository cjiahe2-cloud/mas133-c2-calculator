"""Calculator check (no OpenAI calls, free): uv run python calc_test.py

Expected figures are worked by hand from MAS 133 paras 4.3.5.3-4.3.5.7, 4.1.8 and Appendix 5E paras 16-17B.
Cases without a method use FX_ADJUSTMENT_METHOD's default ("total_less_limit": the FX adjustment is total asset
value less the 40% limit, once FX exposure is over the limit).
"""
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
print(f"\n{len(CASES) - failed} of {len(CASES)} passed.")
raise SystemExit(1 if failed else 0)
