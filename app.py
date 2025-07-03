import streamlit as st
import pandas as pd
import math

# ─── Simulation Function with GAAP Accrual ───────────────────────────────────────
def run_simulation(params):
    months = params["simulation_months"]
    total_pkgs = sum(params["initial_inventory"].values())
    cost_per_pkg = params["initial_inventory_cost"] / total_pkgs

    inventory = {s: int(q) for s, q in params["initial_inventory"].items()}
    cash_balance = -int(params["initial_inventory_cost"])
    pending_orders = []
    monthly_cohorts = []
    prepaid_cohorts = []

    # Month 1 seeds: split into monthly vs prepaid
    init_pre = params["initial_prepaid"]
    init_mon = params["initial_subscribers"] - init_pre
    if init_mon > 0:
        monthly_cohorts.append({"start": 1, "count": init_mon, "stage": 1})
    if init_pre > 0:
        prepaid_cohorts.append({
            "start": 1,
            "count": init_pre,
            "stage": 1,
            "deferred": init_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])
        })

    records = []
    for month in range(1, months + 1):
        # Determine new subs and cash
        if month == 1:
            new_mon, new_pre = init_mon, init_pre
        else:
            alive_prior = sum(c["count"] for c in monthly_cohorts + prepaid_cohorts)
            new_tot = alive_prior * params["subscriber_growth_rate"]
            new_pre = int(round(new_tot * params["percent_prepaid"]))
            new_mon = int(round(new_tot - new_pre))
            # add cohorts
            for st, pct in params["start_stage_dist"].items():
                cnt = int(round(new_mon * pct))
                if cnt > 0:
                    monthly_cohorts.append({"start": month, "count": cnt, "stage": st})
            if new_pre > 0:
                cash_balance += new_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])
                prepaid_cohorts.append({
                    "start": month,
                    "count": new_pre,
                    "stage": 1,
                    "deferred": new_pre * params["monthly_price"] * 9 * (1 - params["prepaid_discount_rate"])
                })

        # Active subscriber counts
        total_active_monthly = sum(c["count"] for c in monthly_cohorts)
        total_active_prepaid = sum(c["count"] for c in prepaid_cohorts)

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
            max_age = (4 - c["stage"]) * 3
            if 1 <= age <= max_age:
                s = min(c["stage"] + (age - 1) // 3, 3)
                ship_mon[s] += c["count"]
                c["count"] = int(round(c["count"] * (1 - params["churn_rate"])))
        for c in prepaid_cohorts:
            age = month - c["start"] + 1
            if 1 <= age <= 9:
                s = min(1 + (age - 1) // 3, 3)
                ship_pre[s] += c["count"]

        # Reorder
        expected = {s: ship_mon[s] + ship_pre[s] for s in (1, 2, 3)}
        inv_cost = 0
        reorder = []
        for s in (1, 2, 3):
            inventory[s] -= expected[s]
            thr = math.ceil(expected[s] * params["reorder_safety"])
            if inventory[s] <= thr:
                pending_orders.append((month + params["lead_time"], s, params["reorder_qty"]))
                inv_cost += params["reorder_cost"]
                reorder.append(s)

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
        rev_total = rev_mon + rev_pre
        cogs_mon = sum(ship_mon.values()) * cost_per_pkg
        total_cogs = cogs_mon + cogs_pre

        cac = sum([new_mon * params["cac_new_monthly"], new_pre * params["cac_new_prepaid"]])
        net = rev_total - cac - total_cogs - inv_cost
        cash_balance += net

        records.append({
            "Month": month,
            "New Monthly Subs": new_mon,
            "New Prepaid Subs": new_pre,
            "Stage 1 To Ship": ship_mon[1] + ship_pre[1],
            "Stage 2 To Ship": ship_mon[2] + ship_pre[2],
            "Stage 3 To Ship": ship_mon[3] + ship_pre[3],
            "Active Monthly Subs": total_active_monthly,
            "Active Prepaid Subs": total_active_prepaid,
            "Monthly Revenue": round(rev_mon, 2),
            "Prepaid Rev Recog": round(rev_pre, 2),
            "Total Revenue": round(rev_total, 2),
            "CAC": round(cac, 2),
            "COGS Mon": round(cogs_mon, 2),
            "COGS Pre": round(cogs_pre, 2),
            "Inv S1": inventory[1],
            "Inv S2": inventory[2],
            "Inv S3": inventory[3],
            "Reorder": reorder,
            "Net Flow": round(net, 2),
            "Cash Balance": round(cash_balance, 2)
        })

    return pd.DataFrame(records)

# ─── Helper: Slider + Input w/ Unique Keys ─────────────────────────────────────
def slider_with_input(label, min_val, max_val, default, step, is_float=False, fmt="%d"):
    col1, col2 = st.sidebar.columns([3, 1])
    key = label.replace(" ", "_")
    if is_float:
        val = col1.slider(label, float(min_val), float(max_val), float(default), float(step), key=key+"_s")
        num = col2.number_input("", float(min_val), float(max_val), value=val, step=float(step), format=fmt, key=key+"_n")
    else:
        val = col1.slider(label, int(min_val), int(max_val), int(default), int(step), key=key+"_s")
        num = col2.number_input("", int(min_val), int(max_val), value=val, step=int(step), format=fmt, key=key+"_n")
    return num

# ─── App Layout ─────────────────────────────────────────────────────────────
st.title("BareBump Cash‑Flow Simulator")
st.sidebar.header("Parameters")

monthly_price = slider_with_input("Sale Price", 0, 500, 75, 1)
init_subs = slider_with_input("Initial Subs", 0, 2000, 250, 10)
init_pre = slider_with_input("Initial Prepaid", 0, 1000, 20, 10)
growth = slider_with_input("Growth Rate", 0.0, 1.0, 0.10, 0.01, True, "%0.2f")
pct_pre = slider_with_input("% Prepaid", 0.0, 1.0, 0.20, 0.01, True, "%0.2f")
disc_pre = slider_with_input("Prepaid Disc", 0.0, 1.0, 0.10, 0.01, True, "%0.2f")
churn = slider_with_input("Churn Rate", 0.0, 1.0, 0.05, 0.01, True, "%0.2f")
lead_time = slider_with_input("Lead Time (# Months)", 0, 6, 1, 1)
safety = slider_with_input("Inv Safety Threshold ×", 1.0, 3.0, 1.2, 0.05, True, "%0.2f")
rqty = slider_with_input("Reorder Qty", 0, 5000, 1330, 10)
rcost = slider_with_input("Reorder Cost", 0, 100000, 25000, 1000)
inv1 = slider_with_input("Inv Stage 1", 0, 5000, 1330, 10)
inv2 = slider_with_input("Inv Stage 2", 0, 5000, 1330, 10)
inv3 = slider_with_input("Inv Stage 3", 0, 5000, 1330, 10)
inv_cost = slider_with_input("Inv Cost", 0, 200000, 75000, 1000)
st1 = slider_with_input("Start S1", 0.0, 1.0, 0.60, 0.01, True, "%0.2f")
st2 = slider_with_input("Start S2", 0.0, 1.0, 0.30, 0.01, True, "%0.2f")
st3 = slider_with_input("Start S3", 0.0, 1.0, 0.10, 0.01, True, "%0.2f")
months = slider_with_input("Months", 1, 36, 12, 1)

params = {
    "monthly_price": monthly_price,
    "initial_subscribers": init_subs,
    "initial_prepaid": init_pre,
    "subscriber_growth_rate": growth,
    "percent_prepaid": pct_pre,
    "prepaid_discount_rate": disc_pre,
    "cac_new_monthly": 20,
    "cac_new_prepaid": 20,
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

df = run_simulation(params).set_index("Month")
st.dataframe(df)


