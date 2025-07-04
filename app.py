import streamlit as st
import pandas as pd
import math

# ─── Define Inputs ─────────────────────────────────────────────────────────────
st.title("BareBump Cash‑Flow Simulator & Financials")
st.sidebar.header("Parameters")

# Input controls
monthly_price   = st.sidebar.number_input("Sale Price", 0, 500, 75)
init_subs       = st.sidebar.number_input("Initial Subs", 0, 1000, 250)
init_pre        = st.sidebar.number_input("Initial Prepaid", 0, 1000, 20)
growth          = st.sidebar.number_input("Growth Rate", 0.0, 1.0, 0.10, format="%.2f")
pct_pre         = st.sidebar.number_input("% Prepaid", 0.0, 1.0, 0.20, format="%.2f")
disc_pre        = st.sidebar.number_input("Prepaid Discount", 0.0, 1.0, 0.10, format="%.2f")
cac_mon         = st.sidebar.number_input("CAC Monthly", 0, 500, 20)
cac_pre         = st.sidebar.number_input("CAC Prepaid", 0, 500, 20)
churn           = st.sidebar.number_input("Churn Rate", 0.0, 1.0, 0.05, format="%.2f")
lead_time       = st.sidebar.number_input("Lead Time (months)", 0, 12, 1)
safety          = st.sidebar.number_input("Safety Factor", 1.0, 3.0, 1.2, format="%.2f")
rqty            = st.sidebar.number_input("Reorder Qty", 0, 5000, 1330)
rcost           = st.sidebar.number_input("Reorder Cost", 0, 100000, 25000)
ship_cost_pkg   = st.sidebar.number_input("Shipping Cost per Package", 0.0, 50.0, 5.0, format="%.2f")
inv1            = st.sidebar.number_input("Inv Stage 1", 0, 5000, 1330)
inv2            = st.sidebar.number_input("Inv Stage 2", 0, 5000, 1330)
inv3            = st.sidebar.number_input("Inv Stage 3", 0, 5000, 1330)
inv_cost        = st.sidebar.number_input("Initial Inv Cost", 0, 500000, 75000)
st1             = st.sidebar.number_input("Start S1 %", 0.0, 1.0, 0.60, format="%.2f")
st2             = st.sidebar.number_input("Start S2 %", 0.0, 1.0, 0.30, format="%.2f")
st3             = st.sidebar.number_input("Start S3 %", 0.0, 1.0, 0.10, format="%.2f")
ship1_1         = st.sidebar.number_input("Pct gets 1 Stage1 shipment", 0.0, 1.0, 0.80, format="%.2f")
ship1_2         = st.sidebar.number_input("Pct gets 2 Stage1 shipments", 0.0, 1.0, 0.15, format="%.2f")
ship1_3         = st.sidebar.number_input("Pct gets 3 Stage1 shipments", 0.0, 1.0, 0.05, format="%.2f")
months          = st.sidebar.number_input("Simulation Months", 1, 36, 12)

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
    "shipping_cost_pkg": ship_cost_pkg,
    "churn_rate": churn,
    "lead_time": lead_time,
    "reorder_safety": safety,
    "start_stage_dist": {1: st1, 2: st2, 3: st3},
    "ship1_dist": {1: ship1_1, 2: ship1_2, 3: ship1_3},
    "simulation_months": months
}

# ─── Simulation Function with GAAP Accrual ───────────────────────────────────────
def run_simulation(params):
    months = params["simulation_months"]
    total_pkgs = sum(params["initial_inventory"].values())
    cost_per_pkg = params["initial_inventory_cost"] / total_pkgs

    # initialize
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
        prepaid_cohorts.append({
            "start": 1, "count": init_pre, "stage": 1,
            "deferred": init_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])
        })
    if init_mon > 0:
        for limit, pct in params["ship1_dist"].items():
            cnt = int(round(init_mon * pct))
            if cnt > 0:
                monthly_cohorts.append({
                    "start": 1, "count": cnt, "stage": 1,
                    "s1_limit": limit, "s1_shipped": 0
                })

    records = []
    for month in range(1, months + 1):
        # New subscriptions
        if month == 1:
            new_mon = sum(c["count"] for c in monthly_cohorts if c.get("start") == 1)
            new_pre = init_pre
        else:
            alive = sum(c["count"] for c in monthly_cohorts + prepaid_cohorts)
            tot = alive * params["subscriber_growth_rate"]
            new_pre = int(round(tot * params["percent_prepaid"]))
            new_mon = int(round(tot - new_pre))
            # apply stage start distribution and ship1 distribution
            for st, spct in params["start_stage_dist"].items():
                for limit, pct in params["ship1_dist"].items():
                    cnt = int(round(new_mon * spct * pct))
                    if cnt > 0:
                        monthly_cohorts.append({
                            "start": month, "count": cnt, "stage": st,
                            "s1_limit": limit, "s1_shipped": 0
                        })
            if new_pre > 0:
                cash_balance += new_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])
                prepaid_cohorts.append({
                    "start": month, "count": new_pre, "stage": 1,
                    "deferred": new_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])
                })

        # Inventory arrivals
        arrivals = [o for o in pending_orders if o[0] == month]
        for _, s, qty in arrivals:
            inventory[s] += qty
        pending_orders = [o for o in pending_orders if o[0] > month]

        # Shipments
        ship_mon = {1: 0, 2: 0, 3: 0}
        ship_pre = {1: 0, 2: 0, 3: 0}
        for c in monthly_cohorts:
            age = month - c["start"] + 1
            if c["stage"] == 1:
                max_age = c["s1_limit"] + 6
            else:
                max_age = (4 - c["stage"]) * 3
            if 1 <= age <= max_age:
                s = min(c["stage"] + (age - 1)//3, 3)
                if s == 1 and c.get("s1_shipped", 0) >= c.get("s1_limit", 3):
                    continue
                ship_mon[s] += c["count"]
                if s == 1:
                    c["s1_shipped"] += 1
                c["count"] = int(round(c["count"] * (1 - params["churn_rate"])))
        for c in prepaid_cohorts:
            age = month - c["start"] + 1
            if 1 <= age <= 9:
                s = min(1 + (age - 1)//3, 3)
                ship_pre[s] += c["count"]

        # Reorder logic
        exp = {s: ship_mon[s] + ship_pre[s] for s in (1, 2, 3)}
        inv_cost = 0
        reorder = []
        for s in (1, 2, 3):
            inventory[s] -= exp[s]
            future_need = sum(exp.values()) * params["lead_time"]
            threshold = math.ceil((exp[s] + future_need) * params["reorder_safety"])
            if inventory[s] <= threshold:
                reorder.append(f"S{s}")
                pending_orders.append((month + params["lead_time"], s, params["reorder_qty"]))
                inv_cost += params["reorder_cost"]

        # Shipping costs
        ship_cost = sum(exp.values()) * params["shipping_cost_pkg"]

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
                cogs_pre += ship_pre[s] * cost_per_pkg
        total_rev = rev_mon + rev_pre
        cogs_mon = sum(ship_mon.values()) * cost_per_pkg
        total_cogs = cogs_mon + cogs_pre
        cac = new_mon * params["cac_new_monthly"] + new_pre * params["cac_new_prepaid"]

        # Financial metrics
        gross = total_rev - total_cogs
        op_inc = gross - cac - ship_cost
        net = op_inc - inv_cost
        cash_balance += net

        # Active subscriber counts
        active_mon = sum(
            c["count"]
            for c in monthly_cohorts
            if 1 <= (month - c["start"] + 1) <= ((4 - c["stage"]) * 3)
        )
        active_pre = sum(
            c["count"]
            for c in prepaid_cohorts
            if 1 <= (month - c["start"] + 1) <= 9
        )

        records.append({
            "Month": month,
            "New Monthly Subs": new_mon,
            "New Prepaid Members": new_pre,
            "Stage 1 To Ship": ship_mon[1] + ship_pre[1],
            "Stage 2 To Ship": ship_mon[2] + ship_pre[2],
            "Stage 3 To Ship": ship_mon[3] + ship_pre[3],
            "Inv S1": inventory[1],
            "Inv S2": inventory[2],
            "Inv S3": inventory[3],
            "Reorder": ",".join(reorder),
            "Active Monthly Subs": active_mon,
            "Active Prepaid Members": active_pre,
            "Monthly Revenue": round(rev_mon, 2),
            "Prepaid Revenue Recog": round(rev_pre, 2),
            "Total Revenue": round(total_rev, 2),
            "Gross Profit": round(gross, 2),
            "Operating Income": round(op_inc, 2),
            "COGS Mon": round(cogs_mon, 2),
            "COGS Pre": round(cogs_pre, 2),
            "Total COGS": round(total_cogs, 2),
            "CAC": round(cac, 2),
            "Shipping Exp": round(ship_cost, 2),
            "Reorder Cost": round(inv_cost, 2),
            "Net Cash Flow": round(net, 2),
            "Cash Balance": round(cash_balance, 2)
        })

    return pd.DataFrame(records).set_index("Month")

# ─── Build Financial Statements ─────────────────────────────────────────────────
def build_financials(df, params):
    # Balance Sheet
    bs = pd.DataFrame({
        "Cash": df["Cash Balance"],
        "Inventory": df[["Inv S1","Inv S2","Inv S3"]].sum(axis=1) * params["initial_inventory_cost"]/sum(params["initial_inventory"].values()),
        "Accounts Recievable": 0,
        "Total Current Assets": lambda x: x["Cash"] + x["Inventory"],
        "PP&E": 0,
        "Goodwill": 0,
        "Total Assets": lambda x: x["Total Current Assets"] + x["PP&E"] + x["Goodwill"],
        "Accounts Payable": 0,
        "Deferred Rev": (df["Prepaid Revenue Recog"].shift(1).fillna(0).cumsum()),
        "Total Current Liabilities": lambda x: x["Accounts Payable"] + x["Deferred Rev"],
        "Long Term Debt": 0,
        "Total Liabilities": lambda x: x["Total Current Liabilities"] + x["Long Term Debt"],
        "Paid-in Capital": params["initial_inventory_cost"],
        "Retained Earnings": df["Operating Income"].cumsum(),
        "Total Equity": lambda x: x["Paid-in Capital"] + x["Retained Earnings"],
        "Total Liab & Equity": lambda x: x["Total Liabilities"] + x["Total Equity"]
    })

    # Income Statement / P&L
    is_df = pd.DataFrame({
        "Revenue": df["Total Revenue"],
        "COGS": df["Total COGS"],
        "Gross Profit": df["Gross Profit"],
        "Operating Expenses": df["CAC"] + df["Shipping Exp"],
        "Operating Income": df["Operating Income"],
        "Other Expenses": 0,
        "Gains": 0,
        "Losses": 0,
        "Net Income": df["Operating Income"]
    })

    # Cash Flow Statement
    cf = pd.DataFrame({
        "Operating Cash Flow": df["Net Cash Flow"],
        "Financing Cash Flow": [params["initial_inventory_cost"]] + [0]*(len(df)-1)
    })

    return bs, is_df, cf

# ─── Run and Display ─────────────────────────────────────────────────────────────
sim_df = run_simulation(data)
bs_df, is_df, cf_df = build_financials(sim_df, data)

st.subheader("Monthly Simulation")
# Format simulation table
sim_disp = sim_df.style.format("{:,}", subset=[col for col in sim_df.columns if sim_df[col].dtype == 'int64'])
sim_disp = sim_disp.format("{:,.2f}", subset=[col for col in sim_df.columns if sim_df[col].dtype == 'float64'])
st.dataframe(sim_disp)

st.subheader("Balance Sheet")
# Format balance sheet
bs_disp = bs_df.style.format("{:,}", subset=[col for col in bs_df.columns if bs_df[col].dtype == 'int64'])
bs_disp = bs_disp.format("{:,.2f}", subset=[col for col in bs_df.columns if bs_df[col].dtype == 'float64'])
st.dataframe(bs_disp)

st.subheader("Income Statement / P&L")
# Format income statement
is_disp = is_df.style.format("{:,}", subset=[col for col in is_df.columns if is_df[col].dtype == 'int64'])
is_disp = is_disp.format("{:,.2f}", subset=[col for col in is_df.columns if is_df[col].dtype == 'float64'])
st.dataframe(is_disp)

st.subheader("Cash Flow Statement")
# Format cash flow statement
cf_disp = cf_df.style.format("{:,}", subset=[col for col in cf_df.columns if cf_df[col].dtype == 'int64'])
cf_disp = cf_disp.format("{:,.2f}", subset=[col for col in cf_df.columns if cf_df[col].dtype == 'float64'])
st.dataframe(cf_disp)
