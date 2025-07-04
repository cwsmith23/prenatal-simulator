import streamlit as st
import pandas as pd
import math

# ─── Define Inputs ─────────────────────────────────────────────────────────────
st.title("BareBump Cash‑Flow Simulator & Financials")
st.sidebar.header("Parameters")

# Input controls
monthly_price = st.sidebar.number_input("Sale Price", 0, 500, 75)
init_subs     = st.sidebar.number_input("Initial Subs", 0, 1000, 250)
init_pre      = st.sidebar.number_input("Initial Prepaid", 0, 1000, 20)
growth        = st.sidebar.number_input("Growth Rate", 0.0, 1.0, 0.10, format="%.2f")
pct_pre       = st.sidebar.number_input("% Prepaid", 0.0, 1.0, 0.20, format="%.2f")
disc_pre      = st.sidebar.number_input("Prepaid Discount", 0.0, 1.0, 0.10, format="%.2f")
cac_mon       = st.sidebar.number_input("CAC Monthly", 0, 500, 20)
cac_pre       = st.sidebar.number_input("CAC Prepaid", 0, 500, 20)
churn         = st.sidebar.number_input("Churn Rate", 0.0, 1.0, 0.05, format="%.2f")
lead_time     = st.sidebar.number_input("Lead Time (months)", 0, 12, 1)
safety        = st.sidebar.number_input("Safety Factor", 1.0, 3.0, 1.2, format="%.2f")
rqty          = st.sidebar.number_input("Reorder Qty", 0, 5000, 1330)
rcost         = st.sidebar.number_input("Reorder Cost", 0, 100000, 25000)
inv1          = st.sidebar.number_input("Inv Stage 1", 0, 5000, 1330)
inv2          = st.sidebar.number_input("Inv Stage 2", 0, 5000, 1330)
inv3          = st.sidebar.number_input("Inv Stage 3", 0, 5000, 1330)
inv_cost      = st.sidebar.number_input("Initial Inv Cost", 0, 500000, 75000)
st1           = st.sidebar.number_input("Start S1 %", 0.0, 1.0, 0.60, format="%.2f")
st2           = st.sidebar.number_input("Start S2 %", 0.0, 1.0, 0.30, format="%.2f")
st3           = st.sidebar.number_input("Start S3 %", 0.0, 1.0, 0.10, format="%.2f")
months        = st.sidebar.number_input("Simulation Months", 1, 36, 12)

# Bundle params
data = {
    "monthly_price": monthly_price,
    "initial_subscribers": init_subs,
    "initial_prepaid": init_pre,
    "subscriber_growth_rate": growth,
    "percent_prepaid": pct_pre,
    "prepaid_discount_rate": disc_pre,
    "cac_new_monthly": cac_mon,
    "cac_new_prepaid": cac_pre,
    "initial_inventory": {1: inv1, 2: inv2, 3: inv3},
    "initial_inventory_cost": inv_cost,
    "reorder_qty": rqty,
    "reorder_cost": rcost,
    "churn_rate": churn,
    "lead_time": lead_time,
    "reorder_safety": safety,
    "start_stage_dist": {1: st1, 2: st2, 3: st3},
    "simulation_months": months
}

# ─── Simulation Function with GAAP Accrual ───────────────────────────────────────
def run_simulation(params):
    months = params["simulation_months"]
    total_pkgs = sum(params["initial_inventory"].values())
    cost_per_pkg = params["initial_inventory_cost"] / total_pkgs

    inventory = {s: int(q) for s, q in params["initial_inventory"].items()}
    cash_balance = 0
    pending_orders = []
    monthly_cohorts = []
    prepaid_cohorts = []

    # Month 1 seeds & cash
    init_pre = params["initial_prepaid"]
    init_mon = params["initial_subscribers"]
    if init_pre > 0:
        cash_balance += init_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])
        prepaid_cohorts.append({"start":1, "count":init_pre, "stage":1,
                                 "deferred": init_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])})
    if init_mon > 0:
        monthly_cohorts.append({"start":1, "count":init_mon, "stage":1})

    records = []
    for month in range(1, months+1):
        # New subscribers
        if month == 1:
            new_mon, new_pre = init_mon, init_pre
        else:
            alive = sum(c["count"] for c in monthly_cohorts + prepaid_cohorts)
            tot = alive * params["subscriber_growth_rate"]
            new_pre = int(round(tot * params["percent_prepaid"]))
            new_mon = int(round(tot - new_pre))
            for st, pct in params["start_stage_dist"].items():
                cnt = int(round(new_mon * pct))
                if cnt > 0:
                    monthly_cohorts.append({"start":month, "count":cnt, "stage":st})
            if new_pre > 0:
                cash_balance += new_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])
                prepaid_cohorts.append({"start":month, "count":new_pre, "stage":1,
                                         "deferred": new_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])})

        # Inventory arrivals
        for _, s, qty in [o for o in pending_orders if o[0] == month]:
            inventory[s] += qty
        pending_orders = [o for o in pending_orders if o[0] > month]

        # Shipments
        ship_mon = {1:0, 2:0, 3:0}
        ship_pre = {1:0, 2:0, 3:0}
        for c in monthly_cohorts:
            age = month - c["start"] + 1
            max_age = (4 - c["stage"]) * 3
            if 1 <= age <= max_age:
                s = min(c["stage"] + (age-1)//3, 3)
                ship_mon[s] += c["count"]
                c["count"] = int(round(c["count"] * (1 - params["churn_rate"])) )
        for c in prepaid_cohorts:
            age = month - c["start"] + 1
            if 1 <= age <= 9:
                s = min(1 + (age-1)//3, 3)
                ship_pre[s] += c["count"]

        # Reorder logic & inventory update
        exp = {s: ship_mon[s] + ship_pre[s] for s in (1,2,3)}
        inv_cost = 0
        reorder = []
        for s in (1,2,3):
            inventory[s] -= exp[s]
            future_need = exp[s] * params["lead_time"]
            threshold = math.ceil((exp[s] + future_need) * params["reorder_safety"])
            if inventory[s] <= threshold:
                reorder.append(f"S{s}")
                pending_orders.append((month + params["lead_time"], s, params["reorder_qty"]))
                inv_cost += params["reorder_cost"]

        # Revenue & COGS accrual
        rev_mon = sum(ship_mon.values()) * params["monthly_price"]
        rev_pre = 0
        cogs_pre = 0
        for c in prepaid_cohorts:
            age = month - c["start"] + 1
            if 1 <= age <= 9:
                slice_rev = c["deferred"] / (10 - age)
                rev_pre += slice_rev
                c["deferred"] -= slice_rev
                cogs_pre += sum(ship_pre.values()) * cost_per_pkg
        total_rev = rev_mon + rev_pre
        cogs_mon = sum(ship_mon.values()) * cost_per_pkg
        total_cogs = cogs_mon + cogs_pre
        cac = new_mon * params["cac_new_monthly"] + new_pre * params["cac_new_prepaid"]

        # Financial metrics
        gross = total_rev - total_cogs
        op_inc = gross - cac
        net = op_inc - inv_cost
        cash_balance += net

        # Active subscribers
        active_mon = sum(c["count"] for c in monthly_cohorts if (month - c["start"] + 1) <= (4 - c["stage"]) * 3)
        active_pre = sum(c["count"] for c in prepaid_cohorts if (month - c["start"] + 1) <= 9)
        deferred = sum(c["deferred"] for c in prepaid_cohorts)

        records.append({
            "Month": month,
            "New Mon": new_mon,
            "New Pre": new_pre,
            "S1 Ship": ship_mon[1] + ship_pre[1],
            "S2 Ship": ship_mon[2] + ship_pre[2],
            "S3 Ship": ship_mon[3] + ship_pre[3],
            "Inv S1": inventory[1],
            "Inv S2": inventory[2],
            "Inv S3": inventory[3],
            "Reorder": reorder,
            "Active Mon": active_mon,
            "Active Pre": active_pre,
            "Rev Mon": round(rev_mon,2),
            "Rev Pre": round(rev_pre,2),
            "Total Rev": round(total_rev,2),
            "CAC": round(cac,2),
            "COGS Mon": round(cogs_mon,2),
            "COGS Pre": round(cogs_pre,2),
            "Gross Profit": round(gross,2),
            "Op Income": round(op_inc,2),
            "Reorder Cost": round(inv_cost,2),
            "Net Flow": round(net,2),
            "Cash Bal": round(cash_balance,2),
            "Deferred": round(deferred,2)
        })

    return pd.DataFrame(records).set_index("Month")

# ─── Financial Statement Builder ───────────────────────────────────────────────
def build_financials(df, params):
    # Balance Sheet
    bs = pd.DataFrame({
        "Cash": df['Cash Bal'],
        "Inventory": df[['Inv S1','Inv S2','Inv S3']].sum(axis=1) * (params['initial_inventory_cost']/sum(params['initial_inventory'].values())),
        "Deferred Rev": df['Deferred'],
        "Paid-in Capital": params['initial_inventory_cost']
    })
    # Income Statement / P&L
    is_df = pd.DataFrame({
        "Revenue": df['Total Rev'],
        "COGS": df['COGS Mon'] + df['COGS Pre'],
        "Operating Expenses": df['CAC'],
        "Net Income": df['Op Income']
    })
    # Cash Flow Statement
    cf = pd.DataFrame({
        "Operating Cash Flow": df['Op Income'],
        "Investing Cash Flow": -df['Reorder Cost'],
        "Net Cash Flow": df['Net Flow']
    })
    return df, bs, is_df.join(cf)

# ─── Run & Display ─────────────────────────────────────────────────────────────
sim_df, bs_df, is_cf_df = build_financials(run_simulation(data), data)
st.subheader("Monthly Simulation")
st.dataframe(sim_df)
st.subheader("Balance Sheet")
st.dataframe(bs_df)
st.subheader("Income Statement & Cash Flow")
st.dataframe(is_cf_df)


