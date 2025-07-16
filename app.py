import streamlit as st
import pandas as pd
import math


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


st.set_page_config(layout="wide")
st.title("BareBump Cash‑Flow Simulator & Financials")

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
rqty          = st.sidebar.number_input("Reorder Quantity (#)", 0, 25000, 1330)
rcost1        = st.sidebar.number_input("Reorder Cost Stage 1 ($)", 0, 1000000, 25000)
rcost2        = st.sidebar.number_input("Reorder Cost Stage 2 ($)", 0, 1000000, 25000)
rcost3        = st.sidebar.number_input("Reorder Cost Stage 3 ($)", 0, 1000000, 25000)
ship_cost_pkg = st.sidebar.number_input("Shipping Cost per Pack ($)", 0.0, 50.0, 5.0, format="%.2f")
inv1          = st.sidebar.number_input("Initial Inv Stage 1 (#)", 0, 25000, 1330)
inv2          = st.sidebar.number_input("Initial Inv Stage 2 (#)", 0, 25000, 1330)
inv3          = st.sidebar.number_input("Initial Inv Stage 3 (#)", 0, 25000, 1330)
inv_cost      = st.sidebar.number_input("Initial Inventory Cost ($)", 0, 500000, 75000)
st1           = st.sidebar.number_input("Start Stage 1 %", 0.0, 1.0, 0.60, format="%.2f")
st2           = st.sidebar.number_input("Start Stage 2 %", 0.0, 1.0, 0.30, format="%.2f")
st3           = st.sidebar.number_input("Start Stage 3 %", 0.0, 1.0, 0.10, format="%.2f")
ship1_1       = st.sidebar.number_input("Pct Ship Stage 1 Initial", 0.0, 1.0, 0.80, format="%.2f")
ship1_2       = st.sidebar.number_input("Pct Ship Stage 2 Initial", 0.0, 1.0, 0.15, format="%.2f")
ship1_3       = st.sidebar.number_input("Pct Ship Stage 3 Initial", 0.0, 1.0, 0.05, format="%.2f")
months        = st.sidebar.number_input("Simulation Months", 1, 36, 12)

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
    "simulation_months":      months
}


def run_simulation(p):
    total_pkgs = sum(p["initial_inventory"].values())
    if total_pkgs == 0:
        raise ValueError("Initial inventory must be greater than zero.")
    inventory_value = p["initial_inventory_cost"]
    inventory       = p["initial_inventory"].copy()
    inventory_transit_value = 0
    monthly_amt  = p["monthly_price"] * (1 - p["prepaid_discount_rate"])
    cash         = p["initial_inventory_cost"]  # financing inflow
    cash        -= p["initial_inventory_cost"]  # initial inventory purchase
    cum_net_cash = 0
    prev_def_bal = 0
    pending      = []
    monthly_cohorts = []
    prepaid_cohorts = []

    # Month 1: seed prepaid
    if p["initial_prepaid"] > 0:
        prepaid_cohorts.append({
            "start":    1,
            "count":    p["initial_prepaid"],
            "deferred": p["initial_prepaid"] * monthly_amt * 9
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
            # seed new prepaid
            if new_pre > 0:
                prepaid_cohorts.append({
                    "start":    m,
                    "count":    new_pre,
                    "deferred": new_pre * monthly_amt * 9
                })

        # Initialize monthly counters
        reorder_cost   = 0
        ship_mon       = {1:0,2:0,3:0}
        ship_pre       = {1:0,2:0,3:0}
        rev_pre        = 0
        reorder_events = []

        # Process inventory arrivals
        arrivals = [x for x in pending if x[0] == m]
        for _, s, qty, cost in arrivals:
            inventory[s] += qty
            inventory_value += cost
            inventory_transit_value -= cost
        pending = [x for x in pending if x[0] > m]

        # Compute current average cost
        tot_inv = sum(inventory.values())
        cost_per_pkg = inventory_value / tot_inv if tot_inv else 0

        # Ship monthly cohorts
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
            ship_mon[s] += c["count"]
            c["count"]  = int(round(c["count"] * (1 - p["churn_rate"])))

        # Ship prepaid cohorts
        for c in prepaid_cohorts:
            age = m - c["start"] + 1
            if 1 <= age <= 9:
                s = min(1 + (age-1)//3, 3)
                ship_pre[s] += c["count"]
                rev_pre     += c["count"] * monthly_amt
                c["deferred"] -= c["count"] * monthly_amt

        # Reorder logic (correct inventory check)
        exp = {s: ship_mon[s] + ship_pre[s] for s in (1, 2, 3)}
        for s in (1, 2, 3):
            # deduct shipments from inventory
            inventory[s] -= exp[s]
            fut = exp[s] * p["lead_time"]
            thr = math.ceil((exp[s] + fut) * p["reorder_safety"])
            if inventory[s] <= thr:
                cost = p["reorder_cost"][s]
                pending.append((m + p["lead_time"], s, p["reorder_qty"], cost))
                reorder_cost += cost
                reorder_events.append(f"S{s}")
                inventory_transit_value += cost

        # Financial calculations
        total_ship = sum(exp.values())
        cogs_mon   = sum(ship_mon.values()) * cost_per_pkg
        cogs_pre   = sum(ship_pre.values()) * cost_per_pkg
        total_cogs = cogs_mon + cogs_pre
        inventory_value -= total_cogs

        rev_mon    = sum(ship_mon.values()) * p["monthly_price"]
        total_rev  = rev_mon + rev_pre
        cac        = new_mon * p["cac_new_monthly"] + new_pre * p["cac_new_prepaid"]
        ship_cost  = total_ship * p["shipping_cost_pkg"]
        gross      = total_rev - total_cogs
        op_inc     = gross - cac
        deferred_bal    = sum(c["deferred"] for c in prepaid_cohorts)
        def_change      = deferred_bal - prev_def_bal
        prev_def_bal    = deferred_bal
        net_inc         = op_inc - ship_cost
        cash_from_sales = total_rev
        cash_expenses   = cac + ship_cost
        net_cash        = cash_from_sales - cash_expenses - reorder_cost + def_change
        cash           += net_cash

        records.append({
            "Month":                  m,
            "New Monthly Subs":       new_mon,
            "New Prepaid Subs":       new_pre,
            "Stage 1 Shipped":        ship_mon[1] + ship_pre[1],
            "Stage 2 Shipped":        ship_mon[2] + ship_pre[2],
            "Stage 3 Shipped":        ship_mon[3] + ship_pre[3],
            "Inv S1":                 inventory[1],
            "Inv S2":                 inventory[2],
            "Inv S3":                 inventory[3],
            "Inventory Value":        round(inventory_value, 2),
            "Transit Value":          round(inventory_transit_value, 2),
            "Reorder Cost":           reorder_cost,
            "Reorder Stages":         ", ".join(reorder_events) or "-",
            "Monthly Revenue":        round(rev_mon,2),
            "Prepaid Rev Recognized": round(rev_pre,2),
            "Total Revenue":          round(total_rev,2),
            "Total COGS":             round(total_cogs,2),
            "Gross Profit":           round(gross,2),
            "Operating Income":       round(op_inc,2),
            "CAC":                    round(cac,2),
            "Shipping Exp":           round(ship_cost,2),
            "Net Income":             round(net_inc,2),
            "Net Cash Flow":          round(net_cash,2),
            "Cash Balance":           round(cash,2),
            "Deferred Rev Balance":   round(deferred_bal,2),
        })

        # Prune expired cohorts
        monthly_cohorts = [
            c for c in monthly_cohorts
            if c["count"]>0 and (m - c["start"] +1) <= c["s1_limit"]+6
        ]
        prepaid_cohorts = [
            c for c in prepaid_cohorts
            if (m - c["start"] +1) < 9
        ]

    return pd.DataFrame(records).set_index("Month")


def build_financials(df, p):
    bs = pd.DataFrame({"Cash Balance": df["Cash Balance"]})
    bs["Inventory Value"]      = df["Inventory Value"] + df["Transit Value"]
    bs["Unearned Revenue"]     = df["Deferred Rev Balance"]
    bs["Total Current Assets"] = bs["Cash Balance"] + bs["Inventory Value"]
    bs["Total Liabilities"]    = bs["Unearned Revenue"]
    bs["Paid‑in Capital"]      = p["initial_inventory_cost"]
    bs["Retained Earnings"]    = df["Net Income"].cumsum()
    bs["Total Equity"]         = bs["Paid‑in Capital"] + bs["Retained Earnings"]
    bs["Total L&E"]            = bs["Total Liabilities"] + bs["Total Equity"]

    # Income Statement Year 1
    is_df = pd.DataFrame({
        "Revenue":      df["Total Revenue"],
        "COGS":         df["Total COGS"],
        "Gross Profit": df["Gross Profit"],
        "Op Expenses":  df["CAC"],
        "Op Income":    df["Operating Income"],
        "Shipping Expenses": df["Shipping Exp"],
        "Net Income":   df["Net Income"],
    })
    annual_is = is_df.head(12).sum().to_frame().T
    annual_is.index = ["Year 1"]

    # Cash Flow Statement
    cf = pd.DataFrame({
        "Operating Cash Flow": df["Net Cash Flow"],
        "Financing Cash Flow": 0
    })
    if not cf.empty:
        cf.iloc[0, cf.columns.get_loc("Financing Cash Flow")] = p["initial_inventory_cost"]

    return bs, annual_is, cf


# ─── Run & Display Reports ────────────────────────────────────────────────────
sim_df = run_simulation(params)
bs_df, annual_is_df, cf_df = build_financials(sim_df, params)

fmt_int = "{:,}"
fmt_flt = "{:,.2f}"

# ─── Prepare & Reorder Monthly Table ─────────────────────────────────────────
# 1) drop Transit Value
# ─── Prepare & Reorder Monthly Table ─────────────────────────────────────────
# 1) drop Transit Value
display_df = sim_df.drop(columns=["Transit Value"])

# 2) specify the exact column order you want
display_cols = [
    "New Monthly Subs", "New Prepaid Subs",
    "Stage 1 Shipped", "Stage 2 Shipped", "Stage 3 Shipped",
    "Inv S1", "Inv S2", "Inv S3",
    "Inventory Value",     # on‑hand inventory
    "Reorder Stages",      # keep stages with reorders
    "Monthly Revenue", "Prepaid Rev Recognized", "Total Revenue",
    "Total COGS", "Gross Profit",
    "CAC",                 # between Gross Profit & Operating Income
    "Operating Income",
    "Shipping Exp",
    "Net Income",          # before Reorder Cost
    "Reorder Cost",        # moved to right after Net Income
    "Net Cash Flow",
    "Cash Balance", "Deferred Rev Balance"
]
display_df = display_df[display_cols]

# 3) render
st.subheader("Monthly Simulation Details")
st.dataframe(
    display_df.style
        .format(fmt_int, subset=display_df.select_dtypes("int").columns)
        .format(fmt_flt, subset=display_df.select_dtypes("float").columns)
)



st.subheader("Annual Income Statement (Year 1)")
st.dataframe(annual_is_df.style.format(fmt_flt))

# ─── 3‑Month Balance Sheet View ───────────────────────────────────────────────
start_month = st.sidebar.number_input(
    "Start Month for 3‑Month View", 1, params["simulation_months"]-2, 1
)
slice_df = bs_df.loc[start_month:start_month+2]
fmt3 = slice_df.T.copy()
fmt3.index = [
    "Cash","Inventory","Unearned Revenue",
    "Total Current Assets","Total Liabilities",
    "Paid‑in Capital","Retained Earnings",
    "Total Equity","Total L&E"
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
for lbl in ["Unearned Revenue","Total Liabilities"]:
    ind = "  " if lbl != "Total Liabilities" else ""
    rows.append((f"{ind}{lbl}", [f"{v:,.2f}" for v in fmt3.loc[lbl]]))
rows.append(("", ["", "", ""]))
# Equity
rows.append(("Shareholders' Equity:", ["", "", ""]))
for lbl in ["Paid‑in Capital","Retained Earnings","Total Equity"]:
    ind = "  " if lbl != "Total Equity" else ""
    rows.append((f"{ind}{lbl}", [f"{v:,.2f}" for v in fmt3.loc[lbl]]))
rows.append(("", ["", "", ""]))
# Total L&E
rows.append(("Total L&E", [f"{v:,.2f}" for v in fmt3.loc["Total L&E"]]))

cols = [f"Month {m}" for m in slice_df.index]
df3 = pd.DataFrame([vals for _, vals in rows], columns=cols)
df3.insert(0, "", [lbl for lbl,_ in rows])

st.subheader("Balance Sheet (3‑Month View)")
st.dataframe(df3, hide_index=True, use_container_width=True)

st.subheader("Balance Sheet (End of Month)")
st.dataframe(bs_df.style.format(fmt_flt))

st.subheader("Monthly Cash Flow Statement")
st.dataframe(cf_df.style.format(fmt_flt))

# Insert this at the bottom of your Streamlit script, after your last st.dataframe() call
with st.expander("📋 All Calculation Methods"):
    st.markdown(r"""
    ### Subscriber Growth and Shipments
    - **Total Active Subscribers**: count of all current monthly and prepaid subscribers
    - **Estimated New Subscribers** = Total Active Subscribers × Monthly Growth Rate
    - **New Prepaid Subscriptions** = Estimated New Subscribers × Percent Prepaid
    - **New Monthly Subscriptions** = Estimated New Subscribers × (1 − Percent Prepaid)
    - **Shipment Breakdown by Stage**:
      - Stage 1 shipment count
      - Stage 2 shipment count
      - Stage 3 shipment count

    ### Inventory Management
    - **End‑of‑Month Inventory for Each Stage** = Previous Month Inventory + New Arrivals − Total Shipments for the Stage
    - **Safety Stock Threshold** = Ceiling of [(Current Month Shipments + (Current Month Shipments × Lead Time in Months)) × Safety Factor]
    - **Reorder Trigger**: when inventory for a stage ≤ Safety Stock Threshold, an order of the Reorder Quantity is placed at the fixed stage reorder cost, arriving after the lead time.

    ### Pricing & Cost of Goods Sold (COGS)
    - **Effective Prepaid Pack Price** = Standard Pack Price × (1 − Prepaid Discount Rate)
    - **Average Cost per Pack** = Total Inventory Value ÷ Total Packs on Hand
    - **COGS** = (Total Monthly Packs Shipped + Total Prepaid Packs Shipped) × Average Cost per Pack

    ### Revenue & Profit
    - **Monthly Subscription Revenue** = Total Monthly Packs Shipped × Standard Pack Price
    - **Prepaid Revenue Recognized** = Total Prepaid Packs Shipped × Effective Prepaid Pack Price
    - **Total Revenue** = Monthly Subscription Revenue + Prepaid Revenue Recognized
    - **Gross Profit** = Total Revenue − COGS

    ### Operating Expenses
    - **Customer Acquisition Cost (CAC)** = (New Monthly Subscriptions × Monthly CAC) + (New Prepaid Subscriptions × Prepaid CAC)
    - **Operating Income** = Gross Profit − Customer Acquisition Cost
    - **Shipping Expense** = (Total Monthly Packs Shipped + Total Prepaid Packs Shipped) × Shipping Cost per Pack
    - **Net Income** = Operating Income − Shipping Expense

    ### Reorder Costs
    - **Reorder Cost** = Sum of fixed reorder fees for each stage reorder event in the month

    ### Cash Flow
    - **Deferred Revenue Balance** = Outstanding prepaid revenue not yet recognized
    - **Change in Deferred Revenue** = This Month’s Deferred Revenue Balance − Last Month’s Deferred Revenue Balance
    - **Cash Inflow from Sales** = Total Revenue
    - **Cash Outflow for Expenses** = Customer Acquisition Cost + Shipping Expense
    - **Net Cash Flow** = Cash Inflow from Sales − Cash Outflow for Expenses − Reorder Cost + Change in Deferred Revenue
    - **Cash Balance** = Previous Cash Balance + Net Cash Flow

    ### Balance Sheet Overview
    - **Cash Balance**: Cumulative cash available at period end
    - **Inventory Value**: Sum of on‑hand pack value and inventory in transit
    - **Unearned Revenue**: Deferred Revenue Balance under liabilities
    - **Paid‑in Capital**: Initial inventory financing amount
    - **Retained Earnings**: Cumulative sum of Net Income over time
    - **Total Equity**: Paid‑in Capital + Retained Earnings
    - **Total Liabilities and Equity**: Unearned Revenue + Total Equity (matches Total Assets)
    """ )
