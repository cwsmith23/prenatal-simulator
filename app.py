import streamlit as st
import pandas as pd
import math
import streamlit.components.v1 as components
from datetime import datetime
import html

# ─── Helpers ──────────────────────────────────────────────────────────────────
def allocate_with_remainder(total, fractions):
    """Allocate integer counts from a total based on fractional weights."""
    allocations = {k: 0 for k in fractions}
    remainders = []
    for k, frac in fractions.items():
        raw = total * frac
        alloc = math.floor(raw)
        allocations[k] = alloc
        remainders.append((raw - alloc, k))
    leftover = total - sum(allocations.values())
    remainders.sort(reverse=True)
    for i in range(leftover):
        _, key = remainders[i]
        allocations[key] += 1
    return allocations

def fmt_val(v):
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)

def dict_to_html_table(d: dict, title_map: dict = None) -> str:
    rows = []
    for k, v in d.items():
        label = title_map.get(k, k) if title_map else k
        rows.append(f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(fmt_val(v))}</td></tr>")
    return "<table><thead><tr><th>Setting</th><th>Value</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

def fmt_nested(d):
    if not isinstance(d, dict):
        return fmt_val(d)
    return ", ".join([f"S{sk}:{fmt_val(sv)}" for sk, sv in d.items()])

def q2(x: float) -> float:
    """Quantize to cents (avoid float drift)."""
    return float(round((x if x is not None else 0.0) + 1e-12, 2))

# ─── App Setup ────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("BareBump Cash-Flow Simulator & Financials (GAAP)")

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
monthly_price = st.sidebar.number_input("Sale Price ($)", 0, 500, 75)
init_subs     = st.sidebar.number_input("Initial Monthly Subs", 0, 100000, 250)
init_pre      = st.sidebar.number_input("Initial Prepaid Subs", 0, 100000, 20)
growth        = st.sidebar.number_input("Growth Rate", 0.0, 1.0, 0.10, format="%.2f")
pct_pre       = st.sidebar.number_input("% Prepaid", 0.0, 1.0, 0.20, format="%.2f")
disc_pre      = st.sidebar.number_input("Prepaid Discount", 0.0, 1.0, 0.10, format="%.2f")
cac_mon       = st.sidebar.number_input("Monthly CAC ($)", 0, 500, 20)
cac_pre       = st.sidebar.number_input("Prepaid CAC ($)", 0, 500, 20)
churn         = st.sidebar.number_input("Churn Rate", 0.0, 1.0, 0.05, format="%.2f")
lead_time     = st.sidebar.number_input("Lead Time (months)", 0, 12, 1)
safety        = st.sidebar.number_input("Safety Factor", 1.0, 3.0, 1.2, format="%.2f")
rqty          = st.sidebar.number_input("Reorder Quantity (#)", 0, 25000, 833)
rcost1        = st.sidebar.number_input("Reorder Cost Stage 1 ($)", 0, 1000000, 32750)
rcost2        = st.sidebar.number_input("Reorder Cost Stage 2 ($)", 0, 1000000, 32750)
rcost3        = st.sidebar.number_input("Reorder Cost Stage 3 ($)", 0, 1000000, 32750)
ship_cost_pkg = st.sidebar.number_input("Shipping Cost per Pack ($)", 0.0, 50.0, 5.0, format="%.2f")
inv1          = st.sidebar.number_input("Initial Inv Stage 1 (#)", 0, 25000, 833)
inv2          = st.sidebar.number_input("Initial Inv Stage 2 (#)", 0, 25000, 833)
inv3          = st.sidebar.number_input("Initial Inv Stage 3 (#)", 0, 25000, 833)
inv_cost      = st.sidebar.number_input("Initial Inventory Cost ($)", 0, 500000, 98250)
st1           = st.sidebar.number_input("Start Stage 1 %", 0.0, 1.0, 0.60, format="%.2f")
st2           = st.sidebar.number_input("Start Stage 2 %", 0.0, 1.0, 0.30, format="%.2f")
st3           = st.sidebar.number_input("Start Stage 3 %", 0.0, 1.0, 0.10, format="%.2f")
ship1_1       = st.sidebar.number_input("Pct Ship Stage 1 Initial", 0.0, 1.0, 0.80, format="%.2f")
ship1_2       = st.sidebar.number_input("Pct Ship Stage 2 Initial", 0.0, 1.0, 0.15, format="%.2f")
ship1_3       = st.sidebar.number_input("Pct Ship Stage 3 Initial", 0.0, 1.0, 0.05, format="%.2f")
months        = st.sidebar.number_input("Simulation Months", 1, 36, 12)
tax_rate      = st.sidebar.slider("Income Tax Rate", 0.0, 1.0, 0.21, step=0.01, format="%.2f")
pay_taxes_now = st.sidebar.checkbox("Pay Taxes Monthly (otherwise accrue to Taxes Payable)", value=False)

params = {
    "monthly_price":          monthly_price,
    "initial_subscribers":    init_subs,
    "initial_prepaid":        init_pre,
    "subscriber_growth_rate": growth,
    "percent_prepaid":        pct_pre,
    "prepaid_discount_rate":  disc_pre,
    "cac_new_monthly":        cac_mon,
    "cac_new_prepaid":        cac_pre,
    "initial_inventory":      {1: inv1, 2: inv2, 3: inv3},
    "initial_inventory_cost": inv_cost,
    "reorder_qty":            rqty,
    "reorder_cost":           {1: rcost1, 2: rcost2, 3: rcost3},
    "shipping_cost_pkg":      ship_cost_pkg,
    "churn_rate":             churn,
    "lead_time":              lead_time,
    "reorder_safety":         safety,
    "start_stage_dist":       {1: st1, 2: st2, 3: st3},
    "ship1_dist":             {1: ship1_1, 2: ship1_2, 3: ship1_3},
    "simulation_months":      months,
}

# Unit cost sanity panel
total_init_units = sum(params["initial_inventory"].values())
init_unit_cost = (params["initial_inventory_cost"] / total_init_units) if total_init_units else 0
reorder_unit_costs = {s: (params["reorder_cost"][s] / params["reorder_qty"] if params["reorder_qty"] else 0)
                      for s in (1,2,3)}
st.caption(
    f"🧮 Unit cost: ${init_unit_cost:,.2f}/pack  "
)

# ─── Core Simulation ─────────────────────────────────────────────────────────
def run_simulation(p):
    total_pkgs = sum(p["initial_inventory"].values())
    if total_pkgs == 0:
        raise ValueError("Initial inventory must be greater than zero.")

    # Inventory & cash
    inventory_value = q2(p["initial_inventory_cost"])       # on-hand value
    inventory       = p["initial_inventory"].copy()         # on-hand units
    inventory_transit_value = q2(0.0)
    monthly_amt  = p["monthly_price"] * (1 - p["prepaid_discount_rate"])
    cash         = q2(p["initial_inventory_cost"])          # equity financing inflow
    cash        = q2(cash - p["initial_inventory_cost"])    # initial inventory purchase
    taxes_payable = q2(0.0)

    # Deferred revenue centralized (exact)
    deferred_bal = q2(p["initial_prepaid"] * monthly_amt * 9)
    prev_def_bal = deferred_bal

    # ⬇️ Add this: you already collected the prepaid cash at t=0
    cash = q2(cash + deferred_bal)

    pending      = []  # (arrive_month, stage, qty, cost)
    monthly_cohorts = []
    prepaid_cohorts = []

    # Month 1: seed prepaid cohort (for shipping cadence only)
    if p["initial_prepaid"] > 0:
        prepaid_cohorts.append({
            "start":    1,
            "count":    p["initial_prepaid"],
        })

    # Month 1: seed monthly cohorts into start stages
    s1_limit = next(iter(p["ship1_dist"].keys()))
    stage_alloc = allocate_with_remainder(p["initial_subscribers"], p["start_stage_dist"])
    for stg, base_cnt in stage_alloc.items():
        if base_cnt == 0:
            continue
        if stg == 1:
            pseudo_start = 1
        elif stg == 2:
            pseudo_start = 1 - (s1_limit + 1)
        else:
            pseudo_start = 1 - (s1_limit + 4)
        ship_alloc = allocate_with_remainder(base_cnt, p["ship1_dist"])
        for lim, cnt in ship_alloc.items():
            if cnt > 0:
                monthly_cohorts.append({
                    "start":    pseudo_start,
                    "count":    cnt,
                    "s1_limit": lim
                })

    records = []
    for m in range(1, p["simulation_months"] + 1):
        # Generate new cohorts after month 1
        if m == 1:
            new_pre = p["initial_prepaid"]
            new_mon = sum(c["count"] for c in monthly_cohorts)
        else:
            alive   = sum(c["count"] for c in monthly_cohorts + prepaid_cohorts)
            tot     = alive * p["subscriber_growth_rate"]
            alloc   = allocate_with_remainder(
                int(round(tot)),
                {"pre": p["percent_prepaid"], "mon": 1 - p["percent_prepaid"]}
            )
            new_pre = alloc["pre"]
            new_mon = alloc["mon"]
            # seed new monthly cohorts
            stage_alloc = allocate_with_remainder(new_mon, p["start_stage_dist"])
            for stg, base_cnt in stage_alloc.items():
                ship_alloc = allocate_with_remainder(base_cnt, p["ship1_dist"])
                for lim, cnt in ship_alloc.items():
                    if cnt > 0:
                        monthly_cohorts.append({
                            "start":    m,
                            "count":    cnt,
                            "s1_limit": lim
                        })
            # seed new prepaid cohort & increase deferred
            if new_pre > 0:
                prepaid_cohorts.append({
                    "start":    m,
                    "count":    new_pre,
                })
                deferred_bal = q2(deferred_bal + q2(new_pre * monthly_amt * 9))

        # Initialize monthly counters
        reorder_cost   = q2(0.0)
        reorder_events = []
        ship_mon_demand = {1:0,2:0,3:0}
        ship_pre_demand = {1:0,2:0,3:0}

        # Process inventory arrivals (capitalize to on-hand; reduce transit)
        arrivals = [x for x in pending if x[0] == m]
        for _, s, qty, cost in arrivals:
            inventory[s] += qty
            inventory_value = q2(inventory_value + cost)
            inventory_transit_value = q2(inventory_transit_value - cost)
        pending = [x for x in pending if x[0] > m]

        # Weighted-average cost per on-hand unit (stabilize to 4 decimals)
        tot_inv = sum(inventory.values())
        cost_per_pkg = round(inventory_value / tot_inv, 4) if tot_inv > 0 else 0.0

        # Ship monthly cohorts → demand by stage
        for c in monthly_cohorts:
            age = m - c["start"] + 1
            if   age <= c["s1_limit"]:
                s = 1
            elif age <= c["s1_limit"] + 3:
                s = 2
            elif age <= c["s1_limit"] + 6:
                s = 3
            else:
                continue
            ship_mon_demand[s] += c["count"]
            c["count"]  = int(round(c["count"] * (1 - p["churn_rate"])))

        # Ship prepaid cohorts → demand by stage
        for c in prepaid_cohorts:
            age = m - c["start"] + 1
            if 1 <= age <= 9:
                s = min(1 + (age-1)//3, 3)
                ship_pre_demand[s] += c["count"]

        # Fill by stage with stockout protection (prepaid priority)
        ship_mon_filled = {1:0,2:0,3:0}
        ship_pre_filled = {1:0,2:0,3:0}
        backorder_mon   = {1:0,2:0,3:0}
        backorder_pre   = {1:0,2:0,3:0}

        for s in (1,2,3):
            available = max(inventory[s], 0)
            filled_pre = min(ship_pre_demand[s], available)
            available -= filled_pre
            filled_mon = min(ship_mon_demand[s], available)
            backorder_pre[s] = ship_pre_demand[s] - filled_pre
            backorder_mon[s] = ship_mon_demand[s] - filled_mon
            ship_pre_filled[s] = filled_pre
            ship_mon_filled[s] = filled_mon
            inventory[s] -= (filled_pre + filled_mon)

        # Reorder logic (use demand to set threshold)
        exp_demand = {s: ship_mon_demand[s] + ship_pre_demand[s] for s in (1,2,3)}
        for s in (1, 2, 3):
            fut = exp_demand[s] * p["lead_time"]
            thr = math.ceil((exp_demand[s] + fut) * p["reorder_safety"])
            if inventory[s] <= thr:
                cost = p["reorder_cost"][s]
                pending.append((m + p["lead_time"], s, p["reorder_qty"], cost))
                reorder_cost = q2(reorder_cost + cost)
                reorder_events.append(f"S{s}")
                inventory_transit_value = q2(inventory_transit_value + cost)

        # Financial calculations (recognize revenue only for filled units)
        total_ship_filled = sum(ship_mon_filled.values()) + sum(ship_pre_filled.values())

        rev_mon    = q2(sum(ship_mon_filled.values()) * p["monthly_price"])
        rev_pre    = q2(sum(ship_pre_filled.values()) * monthly_amt)
        total_rev  = q2(rev_mon + rev_pre)

        cogs_mon   = q2(sum(ship_mon_filled.values()) * cost_per_pkg)
        cogs_pre   = q2(sum(ship_pre_filled.values()) * cost_per_pkg)
        total_cogs = q2(cogs_mon + cogs_pre)
        inventory_value = q2(inventory_value - total_cogs)  # relieve inventory

        shipping_exp  = q2(total_ship_filled * p["shipping_cost_pkg"])
        cac           = q2(new_mon * p["cac_new_monthly"] + new_pre * p["cac_new_prepaid"])
        gross         = q2(total_rev - total_cogs)
        op_expenses   = q2(cac + shipping_exp)
        op_inc        = q2(gross - op_expenses)

        # Deferred revenue: reduce exactly by recognized prepaid revenue
        deferred_bal  = q2(deferred_bal - rev_pre)
        def_change    = q2(deferred_bal - prev_def_bal)
        prev_def_bal  = deferred_bal

        # Taxes
        tax_expense     = q2(max(op_inc, 0) * tax_rate)
        taxes_paid_now  = q2(tax_expense) if pay_taxes_now else q2(0.0)
        taxes_payable   = q2(taxes_payable + tax_expense - taxes_paid_now)
        net_inc_after_tax = q2(op_inc - tax_expense)

        # Operating cash flow: revenue cash (monthly immediate, prepaid via Δdeferred)
        net_cash = q2(total_rev - op_expenses - taxes_paid_now - reorder_cost + def_change)
        cash     = q2(cash + net_cash)

        subscribers_total = sum(c["count"] for c in monthly_cohorts + prepaid_cohorts)
        prepaid_total     = sum(c["count"] for c in prepaid_cohorts)

        records.append({
            "Month":                  m,
            "New Monthly Subs":       new_mon,
            "New Prepaid Subs":       new_pre,
            "Stage 1 Shipped":        ship_mon_filled[1] + ship_pre_filled[1],
            "Stage 2 Shipped":        ship_mon_filled[2] + ship_pre_filled[2],
            "Stage 3 Shipped":        ship_mon_filled[3] + ship_pre_filled[3],
            "Backorder S1":           backorder_mon[1] + backorder_pre[1],
            "Backorder S2":           backorder_mon[2] + backorder_pre[2],
            "Backorder S3":           backorder_mon[3] + backorder_pre[3],
            "Inv S1":                 inventory[1],
            "Inv S2":                 inventory[2],
            "Inv S3":                 inventory[3],
            "Inventory Value":        inventory_value,
            "Transit Value":          inventory_transit_value,
            "Reorder Cost":           reorder_cost,
            "Reorder Stages":         ", ".join(reorder_events) or "-",
            "Monthly Revenue":        rev_mon,
            "Prepaid Rev Recognized": rev_pre,
            "Total Revenue":          total_rev,
            "Total COGS":             total_cogs,
            "Gross Profit":           gross,
            "Operating Expenses":     op_expenses,
            "Operating Income":       op_inc,
            "Tax Expense":            tax_expense,
            "Net Income":             net_inc_after_tax,  # after-tax
            "CAC":                    cac,
            "Shipping Exp":           shipping_exp,
            "Net Cash Flow":          net_cash,
            "Cash Balance":           cash,
            "Taxes Payable":          taxes_payable,
            "Deferred Rev Balance":   deferred_bal,
            "Total Shipments":        total_ship_filled,
            "Total Subscribers":      subscribers_total,
            "Total Prepaid Subs":     prepaid_total,
        })

        # Prune expired cohorts
        monthly_cohorts = [
            c for c in monthly_cohorts
            if c["count"] > 0 and (m - c["start"] + 1) <= c["s1_limit"] + 6
        ]
        prepaid_cohorts = [
            c for c in prepaid_cohorts
            if (m - c["start"] + 1) < 9
        ]

    return pd.DataFrame(records).set_index("Month")

def build_financials(df, p):
    bs = pd.DataFrame({"Cash Balance": df["Cash Balance"]})
    bs["Inventory Value"]      = df["Inventory Value"] + df["Transit Value"]
    bs["Unearned Revenue"]     = df["Deferred Rev Balance"]
    bs["Taxes Payable"]        = df["Taxes Payable"]
    bs["Total Current Assets"] = bs["Cash Balance"] + bs["Inventory Value"]
    bs["Total Liabilities"]    = bs["Unearned Revenue"] + bs["Taxes Payable"]
    bs["Paid-in Capital"]      = p["initial_inventory_cost"]
    bs["Retained Earnings"]    = df["Net Income"].cumsum()           # after-tax
    bs["Total Equity"]         = bs["Paid-in Capital"] + bs["Retained Earnings"]
    bs["Total L&E"]            = bs["Total Liabilities"] + bs["Total Equity"]
    bs["Δ (Assets − L&E)"]     = (bs["Total Current Assets"] - bs["Total L&E"]).round(2)

    # Income Statement (Monthly → Year 1 summary)
    is_df = pd.DataFrame({
        "Revenue":            df["Total Revenue"],
        "COGS":               df["Total COGS"],
        "Gross Profit":       df["Gross Profit"],
        "Operating Expenses": df["Operating Expenses"],  # CAC + Shipping
        "Operating Income":   df["Operating Income"],
        "Tax Expense":        df["Tax Expense"],
        "Net Income":         df["Net Income"],          # after-tax
    })
    annual_is = is_df.head(12).sum().to_frame().T
    annual_is.index = ["Year 1"]

    # Cash Flow Statement (Operating + simple Financing)
    cf = pd.DataFrame({
        "Operating Cash Flow": df["Net Cash Flow"],
        "Financing Cash Flow": 0
    })
    if not cf.empty:
        cf.iloc[0, cf.columns.get_loc("Financing Cash Flow")] = p["initial_inventory_cost"]

    return bs, annual_is, cf

# ─── Run & Display Reports ───────────────────────────────────────────────────
sim_df = run_simulation(params)
bs_df, annual_is_df, cf_df = build_financials(sim_df, params)

fmt_int = "{:,}"
fmt_flt = "{:,.2f}"

# ─── Prepare & Reorder Monthly Table ─────────────────────────────────────────
display_df = sim_df.drop(columns=["Transit Value"])
display_cols = [
    "New Monthly Subs", "New Prepaid Subs",
    "Stage 1 Shipped", "Stage 2 Shipped", "Stage 3 Shipped",
    "Backorder S1", "Backorder S2", "Backorder S3",
    "Total Shipments", "Total Subscribers", "Total Prepaid Subs",
    "Inv S1", "Inv S2", "Inv S3",
    "Inventory Value",
    "Reorder Stages",
    "Monthly Revenue", "Prepaid Rev Recognized", "Total Revenue",
    "Total COGS", "Gross Profit",
    "Operating Expenses",
    "Operating Income",
    "Tax Expense",
    "Net Income",           # after-tax
    "Reorder Cost",
    "Net Cash Flow",
    "Cash Balance", "Deferred Rev Balance", "Taxes Payable"
]
display_df = display_df[display_cols]

st.subheader("Monthly Simulation Details")
st.dataframe(
    display_df.style
        .format(fmt_int, subset=display_df.select_dtypes("int").columns)
        .format(fmt_flt, subset=display_df.select_dtypes("float").columns)
)

# ─── 3-Month Balance Sheet View ──────────────────────────────────────────────
start_month = st.sidebar.number_input(
    "Start Month for 3-Month View", 1, params["simulation_months"]-2, 1
)
slice_df = bs_df.loc[start_month:start_month+2]
fmt3 = slice_df.T.copy()
fmt3.index = [
    "Cash","Inventory","Unearned Revenue","Taxes Payable",
    "Total Current Assets","Total Liabilities",
    "Paid-in Capital","Retained Earnings",
    "Total Equity","Total L&E","Δ (Assets − L&E)"
]

rows = []
# Current Assets
rows.append(("Current Assets:", ["", "", ""]))
for lbl in ["Cash","Inventory","Total Current Assets"]:
    ind = "  " if lbl != "Total Current Assets" else ""
    rows.append((f"{ind}{lbl}", [f"{v:,.2f}" for v in fmt3.loc[lbl]]))
rows.append(("", ["", "", ""]))
# Current Liabilities
rows.append(("Current Liabilities:", ["", "", ""]))
for lbl in ["Unearned Revenue","Taxes Payable","Total Liabilities"]:
    ind = "  " if lbl != "Total Liabilities" else ""
    rows.append((f"{ind}{lbl}", [f"{v:,.2f}" for v in fmt3.loc[lbl]]))
rows.append(("", ["", "", ""]))
# Equity
rows.append(("Shareholders' Equity:", ["", "", ""]))
for lbl in ["Paid-in Capital","Retained Earnings","Total Equity"]:
    ind = "  " if lbl != "Total Equity" else ""
    rows.append((f"{ind}{lbl}", [f"{v:,.2f}" for v in fmt3.loc[lbl]]))
rows.append(("", ["", "", ""]))
# Total L&E + Check
rows.append(("Total L&E", [f"{v:,.2f}" for v in fmt3.loc["Total L&E"]]))
rows.append(("Assets − L&E", [f"{v:,.2f}" for v in (fmt3.loc["Total Current Assets"] - fmt3.loc["Total L&E"])]))

cols = [f"Month {m}" for m in slice_df.index]
df3 = pd.DataFrame([vals for _, vals in rows], columns=cols)
df3.insert(0, "", [lbl for lbl,_ in rows])

st.subheader("Balance Sheet (3-Month View) — Change Starting Month in Sidebar")
height_px = (len(df3) + 1) * 35
st.dataframe(df3, hide_index=True, use_container_width=True, height=height_px)

st.subheader("Annual Income Statement")
st.dataframe(annual_is_df.style.format(fmt_flt))

with st.expander("📊 Balance Sheet (Months 1-12)"):
    bs_order = [
        "Cash Balance",
        "Inventory Value",
        "Total Current Assets",
        "Unearned Revenue",
        "Taxes Payable",
        "Total Liabilities",
        "Paid-in Capital",
        "Retained Earnings",
        "Total Equity",
        "Total L&E",
        "Δ (Assets − L&E)"
    ]
    st.dataframe(
        bs_df[bs_order]
            .style
            .format(fmt_flt)
    )

with st.expander("📈 Monthly Cash Flow Statement"):
    st.dataframe(
        cf_df.style
             .format(fmt_flt)
    )

# All calculation methods (updated)
with st.expander("📋 All Calculation Methods"):
    st.markdown(r"""
    ### Revenue Recognition
    - **Monthly subscriptions**: revenue recognized when packs ship (no ship → no revenue).
    - **Prepaid subscriptions**: cash received upfront (flows via **+ΔDeferred**); revenue recognized monthly as shipped; **Deferred Revenue** reduced by prepaid revenue recognized this month.

    ### Inventory & COGS
    - **Inventory**: capitalized at cost when ordered (in-transit), then moved on-hand on arrival.
    - **COGS**: weighted-average cost per on-hand pack × packs actually shipped. Average cost stabilized to 4 decimals.

    ### Operating Expenses
    - **CAC** expensed as incurred.
    - **Outbound Shipping** included in Operating Expenses (not COGS).

    ### Taxes
    - **Tax Expense** = max(Operating Income, 0) × tax rate; accrued monthly.
    - **Taxes Payable** increases by Tax Expense and decreases by any taxes paid (if enabled).

    ### Cash Flow (Operating)
    - **Operating Cash Flow** = Monthly revenue cash − CAC − Shipping − Inventory purchases − Taxes Paid + **ΔDeferred Revenue**.
    - Reorders reduce cash when ordered; inventory value increases when arriving (in-transit → on-hand).

    ### Equity & Retained Earnings
    - **Paid-in Capital**: initial financing.
    - **Retained Earnings**: cumulative **after-tax** income.

    ### Balance Sheet Identity
    - **Assets** = Cash + Inventory (on-hand + in-transit).
    - **Liabilities** = Unearned Revenue + Taxes Payable.
    - **Equity** = Paid-in Capital + Retained Earnings.
    - Check column: **Δ(Assets − L&E)** should be 0.00.
    """)
# ─── Quick Print & Download (Annual IS first, then 12-month BS; taller rows + 1/8" margins; hide button on print) ───
settings_map = {
    "monthly_price":          "Sale Price ($)",
    "initial_subscribers":    "Initial Monthly Subs",
    "initial_prepaid":        "Initial Prepaid Subs",
    "subscriber_growth_rate": "Growth Rate",
    "percent_prepaid":        "% Prepaid",
    "prepaid_discount_rate":  "Prepaid Discount",
    "cac_new_monthly":        "Monthly CAC ($)",
    "cac_new_prepaid":        "Prepaid CAC ($)",
    "churn_rate":             "Churn Rate",
    "lead_time":              "Lead Time (months)",
    "reorder_safety":         "Safety Factor",
    "reorder_qty":            "Reorder Quantity (#)",
    "shipping_cost_pkg":      "Shipping Cost per Pack ($)",
    "initial_inventory":      "Initial Inventory by Stage",
    "initial_inventory_cost": "Initial Inventory Cost ($)",
    "reorder_cost":           "Reorder Cost by Stage",
    "start_stage_dist":       "Start Stage %",
    "ship1_dist":             "Pct Ship Stage 1 Initial",
    "simulation_months":      "Simulation Months",
}
_settings = {k: params.get(k) for k in settings_map.keys()}
for k in ("initial_inventory", "reorder_cost", "start_stage_dist", "ship1_dist"):
    _settings[k] = fmt_nested(_settings[k])
_settings["tax_rate"] = f"{tax_rate:.2%}"
_settings["pay_taxes_monthly"] = "Yes" if pay_taxes_now else "No"
_settings["start_month_3mo_view"] = start_month

settings_html = dict_to_html_table(_settings, settings_map | {
    "tax_rate": "Income Tax Rate",
    "pay_taxes_monthly": "Pay Taxes Monthly?",
    "start_month_3mo_view": "Start Month (3-Month BS View)"
})

# ---------- Build HTML sections (commas kept; no clipping) ----------
def _fmt_int(x):
    try:
        return "" if pd.isna(x) else f"{int(x):,}"
    except Exception:
        return x

def _fmt_flt(x):
    try:
        return "" if pd.isna(x) else f"{x:,.2f}"
    except Exception:
        return x

# Monthly Simulation Details (wrapped so we can scale for print only)
_int_cols  = display_df.select_dtypes(include=["int", "int64"]).columns
_flt_cols  = display_df.select_dtypes(include=["float", "float64"]).columns
_monthly_formatters = {**{c: _fmt_int for c in _int_cols}, **{c: _fmt_flt for c in _flt_cols}}
monthly_html_core = display_df.to_html(index=True, border=0, formatters=_monthly_formatters)
monthly_html = f'<div class="monthly-wrap">{monthly_html_core}</div>'

# Annual Income Statement
_annual_formatters = {c: _fmt_flt for c in annual_is_df.columns}
annual_is_html = annual_is_df.to_html(index=True, border=0, formatters=_annual_formatters)

# 12-month Balance Sheet
bs12_order = [
    "Cash Balance",
    "Inventory Value",
    "Total Current Assets",
    "Unearned Revenue",
    "Taxes Payable",
    "Total Liabilities",
    "Paid-in Capital",
    "Retained Earnings",
    "Total Equity",
    "Total L&E",
    "Δ (Assets − L&E)"
]
_bs12 = bs_df[bs12_order]
_bs12_formatters = {c: _fmt_flt for c in _bs12.columns}
bs12_html = _bs12.to_html(index=True, border=0, formatters=_bs12_formatters)

# 1/8" print margins + landscape for width; ensure monthly table fits fully
force_landscape = True
page_size_css = "@page { size: Letter landscape; margin: 0.125in; }" if force_landscape else "@page { margin: 0.125in; }"

generated_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
print_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BareBump – Quick Report</title>
  <style>
    {page_size_css}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: Arial, sans-serif;
      padding: 24px;              /* on-screen padding only */
      color: #111;
      font-size: 14px;
    }}
    h1 {{ font-size: 26px; margin-bottom: 6px; }}
    h2 {{ font-size: 20px; margin-top: 24px; margin-bottom: 10px; }}

    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 8px 0 24px;
      table-layout: auto;         /* natural widths so numbers aren't cut */
      font-size: 13px;
      max-width: none;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 12px 10px;         /* taller rows */
      line-height: 1.5;           /* more vertical space */
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;        /* keep numbers on one line */
      overflow: visible;
      text-overflow: clip;
    }}
    th {{ background: #f6f6f6; font-weight: bold; white-space: normal; }}

    /* Make sure the Monthly table always fits on page width when printing */
    .monthly-wrap table {{ font-size: 12.25px; }}

    tr {{ page-break-inside: avoid; }}
    .pagebreak {{ page-break-before: always; }}

    @media print {{
      body {{ padding: 0; }}      /* rely on @page margins when printing */
      .noprint {{ display: none !important; }}  /* ← hide the print button on paper */
      table {{ font-size: 12.25px; }}
      th, td {{ padding: 10px 8px; line-height: 1.45; }}
      .monthly-wrap {{
        /* Slight downscale just for the Monthly table to guarantee full width */
        transform: scale(0.96);
        transform-origin: top left;
        width: 104%; /* counteract scale so table remains crisp and centered */
      }}
      .monthly-wrap table {{ font-size: 11.25px; }}
    }}

    .header {{
      display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 8px;
    }}
    .small {{ font-size: 12px; color: #555; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>BareBump – Quick Report</h1>
    <div class="small">Generated: {generated_ts}</div>
  </div>

  <h2>Simulation Settings</h2>
  {settings_html}

  <div class="pagebreak"></div>
  <h2>Monthly Simulation Details</h2>
  {monthly_html}

  <div class="pagebreak"></div>
  <h2>Annual Income Statement</h2>
  {annual_is_html}

  <div class="pagebreak"></div>
  <h2>Balance Sheet (Months 1–12)</h2>
  {bs12_html}

  <div class="noprint" style="margin-top:16px;">
    <button onclick="window.print()">Print</button>
  </div>
</body>
</html>"""

# Button and download
if st.button("🖨️ Quick Print (Settings + Monthly + Annual IS + 12-Mo BS)"):
    components.html(
        f"""
        <script>
        const html = `{print_doc.replace("`","\\`")}`;
        const w = window.open("", "_blank");
        w.document.write(html);
        w.document.close();
        w.focus();
        w.print();
        </script>
        """,
        height=0, width=0
    )

st.download_button(
    "⬇️ Download Quick Report (HTML)",
    data=print_doc.encode("utf-8"),
    file_name="BareBump_Quick_Report.html",
    mime="text/html"
)
