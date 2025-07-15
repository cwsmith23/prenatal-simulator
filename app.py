import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(layout="wide")
st.title("BareBump Cash‑Flow Simulator & Financials")

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
monthly_price = st.sidebar.number_input("Sale Price ($)", 0, 500, 75)
init_subs     = st.sidebar.number_input("Initial Monthly Subs", 0, 1000, 250)
init_pre      = st.sidebar.number_input("Initial Prepaid Subs", 0, 1000, 20)
growth        = st.sidebar.number_input("Growth Rate", 0.0, 1.0, 0.10, format="%.2f")
pct_pre       = st.sidebar.number_input("% Prepaid", 0.0, 1.0, 0.20, format="%.2f")
disc_pre      = st.sidebar.number_input("Prepaid Discount", 0.0, 1.0, 0.10, format="%.2f")
cac_mon       = st.sidebar.number_input("Monthly CAC ($)", 0, 500, 20)
cac_pre       = st.sidebar.number_input("Prepaid CAC ($)", 0, 500, 20)
churn         = st.sidebar.number_input("Churn Rate", 0.0, 1.0, 0.05, format="%.2f")
lead_time     = st.sidebar.number_input("Lead Time (months)", 0, 12, 1)
safety        = st.sidebar.number_input("Safety Factor", 1.0, 3.0, 1.2, format="%.2f")
rqty          = st.sidebar.number_input("Reorder Quantity (#)", 0, 5000, 1330)
rcost         = st.sidebar.number_input("Reorder Cost ($)", 0, 100000, 25000)
ship_cost_pkg = st.sidebar.number_input("Shipping Cost per Pack ($)", 0.0, 50.0, 5.0, format="%.2f")
inv1          = st.sidebar.number_input("Initial Inv Stage 1 (#)", 0, 5000, 1330)
inv2          = st.sidebar.number_input("Initial Inv Stage 2 (#)", 0, 5000, 1330)
inv3          = st.sidebar.number_input("Initial Inv Stage 3 (#)", 0, 5000, 1330)
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
    "reorder_cost":           rcost,
    "shipping_cost_pkg":      ship_cost_pkg,
    "churn_rate":             churn,
    "lead_time":              lead_time,
    "reorder_safety":         safety,
    "start_stage_dist":       {1: st1, 2: st2, 3: st3},
    "ship1_dist":             {1: ship1_1, 2: ship1_2, 3: ship1_3},
    "simulation_months":      months
}

# ─── Simulation ───────────────────────────────────────────────────────────────
def run_simulation(p):
    total_pkgs   = sum(p["initial_inventory"].values())
    cost_per_pkg = p["initial_inventory_cost"] / total_pkgs
    inventory      = {s: int(q) for s, q in p["initial_inventory"].items()}
    monthly_amt    = p["monthly_price"] * (1 - p["prepaid_discount_rate"])
    cash           = 0
    pending        = []
    monthly_cohorts = []
    prepaid_cohorts = []

    # Month 1 seeds & deferred
    if p["initial_prepaid"] > 0:
        cash += p["initial_prepaid"] * monthly_amt * 9
        prepaid_cohorts.append({
            "start": 1,
            "count": p["initial_prepaid"],
            "deferred": p["initial_prepaid"] * monthly_amt * 9,
        })
    if p["initial_subscribers"] > 0:
        for lim, pct in p["ship1_dist"].items():
            cnt = int(round(p["initial_subscribers"] * pct))
            if cnt:
                monthly_cohorts.append({"start": 1, "count": cnt, "s1_limit": lim})

    records = []
    for m in range(1, p["simulation_months"] + 1):
        # New cohorts
        if m == 1:
            new_mon = sum(c["count"] for c in monthly_cohorts)
            new_pre = p["initial_prepaid"]
        else:
            alive = sum(c["count"] for c in monthly_cohorts + prepaid_cohorts)
            tot = alive * p["subscriber_growth_rate"]
            new_pre = int(round(tot * p["percent_prepaid"]))
            new_mon = int(round(tot - new_pre))
            for stg, spct in p["start_stage_dist"].items():
                for lim, pct in p["ship1_dist"].items():
                    cnt = int(round(new_mon * spct * pct))
                    if cnt:
                        monthly_cohorts.append({"start": m, "count": cnt, "s1_limit": lim})
            if new_pre:
                cash += new_pre * monthly_amt * 9
                prepaid_cohorts.append({
                    "start": m,
                    "count": new_pre,
                    "deferred": new_pre * monthly_amt * 9,
                })

        # Arrivals & inventory cost
        inv_cost = 0
        for arr in [x for x in pending if x[0] == m]:
            _, s, qty, cost = arr
            inventory[s] += qty
            inv_cost += cost
        pending = [x for x in pending if x[0] > m]

        # Shipments
        ship_mon = {1: 0, 2: 0, 3: 0}
        ship_pre = {1: 0, 2: 0, 3: 0}
        for c in monthly_cohorts:
            age = m - c["start"] + 1
            lim = c["s1_limit"]
            if age <= lim:
                s = 1
            elif age <= lim + 3:
                s = 2
            elif age <= lim + 6:
                s = 3
            else:
                continue
            ship_mon[s] += c["count"]
            c["count"] = int(round(c["count"] * (1 - p["churn_rate"])))
        rev_pre = 0
        for c in prepaid_cohorts:
            age = m - c["start"] + 1
            if 1 <= age <= 9:
                s = min(1 + (age - 1) // 3, 3)
                ship_pre[s] += c["count"]
                rev_pre += c["count"] * monthly_amt
                c["deferred"] -= c["count"] * monthly_amt

        # Reorder logic
        exp = {s: ship_mon[s] + ship_pre[s] for s in (1, 2, 3)}
        reorder_cost = 0
        for s in (1, 2, 3):
            inventory[s] -= exp[s]
            fut = exp[s] * p["lead_time"]
            thr = math.ceil((exp[s] + fut) * p["reorder_safety"])
            if inventory[s] <= thr:
                pending.append((m + p["lead_time"], s, p["reorder_qty"], p["reorder_cost"]))
                reorder_cost += p["reorder_cost"]

        # Financials
        rev_mon = sum(ship_mon.values()) * p["monthly_price"]
        cogs_mon = sum(ship_mon.values()) * cost_per_pkg
        cogs_pre = sum(ship_pre.values()) * cost_per_pkg
        total_rev, total_cogs = rev_mon + rev_pre, cogs_mon + cogs_pre

        cac = new_mon * p["cac_new_monthly"] + new_pre * p["cac_new_prepaid"]
        ship_cost = sum(exp.values()) * p["shipping_cost_pkg"]
        gross = total_rev - total_cogs
        op_inc = gross - cac
        net_inc = op_inc - ship_cost

        net = rev_mon - cac - ship_cost - inv_cost - reorder_cost
        cash += net
        deferred_bal = sum(c["deferred"] for c in prepaid_cohorts)

        records.append({
            "Month": m,
            "New Monthly Subs": new_mon,
            "New Prepaid Subs": new_pre,
            "Stage 1 Shipped": ship_mon[1] + ship_pre[1],
            "Stage 2 Shipped": ship_mon[2] + ship_pre[2],
            "Stage 3 Shipped": ship_mon[3] + ship_pre[3],
            "Inv S1": inventory[1],
            "Inv S2": inventory[2],
            "Inv S3": inventory[3],
            "Reorder Cost": reorder_cost,
            "Monthly Revenue": round(rev_mon, 2),
            "Prepaid Rev Recognized": round(rev_pre, 2),
            "Total COGS": round(total_cogs, 2),
            "Total Revenue": round(total_rev, 2),
            "Gross Profit": round(gross, 2),
            "Operating Income": round(op_inc, 2),
            "CAC": round(cac, 2),
            "Shipping Exp": round(ship_cost, 2),
            "Net Income": round(net_inc, 2),
            "Net Cash Flow": round(net, 2),
            "Cash Balance": round(cash, 2),
            "Deferred Rev Balance": round(deferred_bal, 2),
        })

        # Remove expired cohorts for next month's calculations
        monthly_cohorts = [
            c for c in monthly_cohorts
            if (m - c["start"] + 1) <= c["s1_limit"] + 6 and c["count"] > 0
        ]
        prepaid_cohorts = [
            c for c in prepaid_cohorts
            if (m - c["start"] + 1) < 9
        ]

    return pd.DataFrame(records).set_index("Month")


def build_financials(df, p):
    total_pkgs = sum(p["initial_inventory"].values())
    bs = pd.DataFrame({"Cash Balance": df["Cash Balance"]})
    bs["Inventory Value"]      = df[["Inv S1", "Inv S2", "Inv S3"]].sum(axis=1) * p["initial_inventory_cost"] / total_pkgs
    bs["Unearned Revenue"]     = df["Deferred Rev Balance"]
    bs["Total Current Assets"] = bs["Cash Balance"] + bs["Inventory Value"]
    bs["Total Liabilities"]    = bs["Unearned Revenue"]
    bs["Paid‑in Capital"]      = p["initial_inventory_cost"]
    bs["Retained Earnings"]    = df["Net Income"].cumsum()
    bs["Total Equity"]         = bs["Paid‑in Capital"] + bs["Retained Earnings"]
    bs["Total L&E"]            = bs["Total Liabilities"] + bs["Total Equity"]

    # Annual Income Statement
    is_df = pd.DataFrame({
        "Revenue":      df["Total Revenue"],
        "COGS":         df["Total COGS"],
        "Gross Profit": df["Gross Profit"],
        "Op Expenses":  df["CAC"] + df["Shipping Exp"],
        "Op Income":    df["Operating Income"] - df["Shipping Exp"],
        "Net Income":   df["Net Income"],
    })
    annual_is = is_df.head(12).sum().to_frame().T
    annual_is.index = ["Year 1"]

    # Monthly Cash Flow Statement
    cf = pd.DataFrame({
        "Operating Cash Flow": df["Net Cash Flow"],  # already includes reorder cost
        "Financing Cash Flow": pd.Series(0, index=df.index)
    })
    cf.iloc[0, cf.columns.get_loc("Financing Cash Flow")] -= p["initial_inventory_cost"]

    return bs, annual_is, cf


# ─── Run & Display Standard Reports ───────────────────────────────────────────
sim_df, bs_df, annual_is_df, cf_df = None, None, None, None
sim_df = run_simulation(params)
bs_df, annual_is_df, cf_df = build_financials(sim_df, params)

fmt_int = "{:,}"
fmt_flt = "{:,.2f}"

st.subheader("Monthly Simulation Details")
st.dataframe(
    sim_df.style
          .format(fmt_int, subset=sim_df.select_dtypes("int").columns)
          .format(fmt_flt, subset=sim_df.select_dtypes("float").columns)
)

st.subheader("Balance Sheet (End of Month)")
st.dataframe(bs_df.style.format(fmt_flt))

st.subheader("Annual Income Statement (Year 1)")
st.dataframe(annual_is_df.style.format(fmt_flt))

st.subheader("Monthly Cash Flow Statement")
st.dataframe(cf_df.style.format(fmt_flt))

# ─── 3‑Month Balance Sheet View ───────────────────────────────────────────────
start_month = st.sidebar.number_input(
    "Start Month for 3‑Month Balance Sheet", 1, params["simulation_months"] - 2, 1
)
slice_df = bs_df.loc[start_month : start_month + 2]
fmt3 = slice_df.T.copy()
fmt3.index = [
    "Cash", "Inventory", "Current Assets",
    "Unearned Revenue", "Liabilities",
    "Paid‑in Capital", "Retained Earnings",
    "Total Equity", "Total L&E",
]

rows = []
rows.append(("Current Assets:", ["", "", ""]))
for lbl in ["Cash", "Inventory", "Current Assets"]:
    rows.append((lbl, [f"{v:,.2f}" for v in fmt3.loc[lbl]]))
rows.append(("", ["", "", ""]))

rows.append(("Current Liabilities:", ["", "", ""]))
for lbl in ["Unearned Revenue", "Liabilities"]:
    rows.append((lbl, [f"{v:,.2f}" for v in fmt3.loc[lbl]]))
rows.append(("", ["", "", ""]))

rows.append(("Shareholders' Equity:", ["", "", ""]))
for lbl in ["Paid‑in Capital", "Retained Earnings", "Total Equity"]:
    rows.append((lbl, [f"{v:,.2f}" for v in fmt3.loc[lbl]]))
rows.append(("", ["", "", ""]))

rows.append(("Total L&E", [f"{v:,.2f}" for v in fmt3.loc["Total L&E"]]))

cols = [f"Month {m}" for m in slice_df.index]
df3 = pd.DataFrame([vals for _, vals in rows], columns=cols)
df3.insert(0, "", [lbl for lbl, _ in rows])

st.subheader("Balance Sheet (3‑Month View)")
st.dataframe(df3, hide_index=True, use_container_width=True)
