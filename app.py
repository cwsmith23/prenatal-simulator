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
    "monthly_price":           monthly_price,
    "initial_subscribers":     init_subs,
    "initial_prepaid":         init_pre,
    "subscriber_growth_rate":  growth,
    "percent_prepaid":         pct_pre,
    "prepaid_discount_rate":   disc_pre,
    "cac_new_monthly":         cac_mon,
    "cac_new_prepaid":         cac_pre,
    "initial_inventory":       {1: inv1, 2: inv2, 3: inv3},
    "initial_inventory_cost":  inv_cost,
    "reorder_qty":             rqty,
    "reorder_cost":            rcost,
    "shipping_cost_pkg":       ship_cost_pkg,
    "churn_rate":              churn,
    "lead_time":               lead_time,
    "reorder_safety":          safety,
    "start_stage_dist":        {1: st1, 2: st2, 3: st3},
    "ship1_dist":              {1: ship1_1, 2: ship1_2, 3: ship1_3},
    "simulation_months":       months
}

# ─── Simulation Function with GAAP Accrual ───────────────────────────────────────
def run_simulation(params):
    months       = params["simulation_months"]
    total_pkgs   = sum(params["initial_inventory"].values())
    cost_per_pkg = params["initial_inventory_cost"] / total_pkgs

    inventory      = {s: int(q) for s, q in params["initial_inventory"].items()}
    cash_balance   = 0
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
            alive  = sum(c["count"] for c in monthly_cohorts + prepaid_cohorts)
            tot    = alive * params["subscriber_growth_rate"]
            new_pre = int(round(tot * params["percent_prepaid"]))
            new_mon  = int(round(tot - new_pre))

            # allocate new monthly cohorts
            for stg, spct in params["start_stage_dist"].items():
                for limit, pct in params["ship1_dist"].items():
                    cnt = int(round(new_mon * spct * pct))
                    if cnt > 0:
                        monthly_cohorts.append({
                            "start": month, "count": cnt, "stage": stg,
                            "s1_limit": limit, "s1_shipped": 0
                        })
            if new_pre > 0:
                cash_balance += new_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])
                prepaid_cohorts.append({
                    "start": month, "count": new_pre, "stage": 1,
                    "deferred": new_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])\
