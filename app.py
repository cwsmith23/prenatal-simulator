import streamlit as st
import pandas as pd
import math

# ─── Sidebar Inputs ────────────────────────────────────────────────────────────
st.title("BareBump Cash‑Flow Simulator & Financials")
st.sidebar.header("Parameters")

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
ship_cost_pkg = st.sidebar.number_input("Shipping Cost/pkg", 0.0, 50.0, 5.0, format="%.2f")
inv1          = st.sidebar.number_input("Inv Stage 1", 0, 5000, 1330)
inv2          = st.sidebar.number_input("Inv Stage 2", 0, 5000, 1330)
inv3          = st.sidebar.number_input("Inv Stage 3", 0, 5000, 1330)
inv_cost      = st.sidebar.number_input("Initial Inv Cost", 0, 500000, 75000)
st1           = st.sidebar.number_input("Start S1 %", 0.0, 1.0, 0.60, format="%.2f")
st2           = st.sidebar.number_input("Start S2 %", 0.0, 1.0, 0.30, format="%.2f")
st3           = st.sidebar.number_input("Start S3 %", 0.0, 1.0, 0.10, format="%.2f")
ship1_1       = st.sidebar.number_input("Pct gets 1 S1", 0.0, 1.0, 0.80, format="%.2f")
ship1_2       = st.sidebar.number_input("Pct gets 2 S1", 0.0, 1.0, 0.15, format="%.2f")
ship1_3       = st.sidebar.number_input("Pct gets 3 S1", 0.0, 1.0, 0.05, format="%.2f")
months        = st.sidebar.number_input("Months", 1, 36, 12)

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

def run_simulation(p):
    months = p["simulation_months"]
    total_pkgs = sum(p["initial_inventory"].values())
    cost_per_pkg = p["initial_inventory_cost"] / total_pkgs

    inventory = {s: int(q) for s, q in p["initial_inventory"].items()}
    cash = 0
    pending = []
    monthly_cohorts = []
    prepaid_cohorts = []

    # Month 1 seeds & deferred cash
    init_pre = p["initial_prepaid"]
    init_mon = p["initial_subscribers"]
    if init_pre > 0:
        cash += init_pre * p["monthly_price"] * 9 * (1 - p["prepaid_discount_rate"])
        prepaid_cohorts.append({
            "start": 1,
            "count": init_pre,
            "stage": 1,
            "deferred": init_pre * p["monthly_price"] * 9 * (1 - p["prepaid_discount_rate"])
        })
    if init_mon > 0:
        for lim, pct in p["ship1_dist"].items():
            cnt = int(round(init_mon * pct))
            if cnt > 0:
                monthly_cohorts.append({
                    "start": 1,
                    "count": cnt,
                    "stage": 1,
                    "s1_limit": lim,
                    "s1_shipped": 0
                })

    records = []
    for m in range(1, months + 1):
        # New subs
        if m == 1:
            new_mon = sum(c["count"] for c in monthly_cohorts)
            new_pre = init_pre
        else:
            alive = sum(c["count"] for c in monthly_cohorts + prepaid_cohorts)
            tot = alive * p["subscriber_growth_rate"]
            new_pre = int(round(tot * p["percent_prepaid"]))
            new_mon = int(round(tot - new_pre))
            # allocate new cohorts
            for stg, spct in p["start_stage_dist"].items():
                for lim, pct in p["ship1_dist"].items():
                    cnt = int(round(new_mon * spct * pct))
                    if cnt > 0:
                        monthly_cohorts.append({"start": m, "count": cnt, "stage": stg, "s1_limit": lim, "s1_shipped": 0})
            if new_pre > 0:
                cash += new_pre * p["monthly_price"] * 9 * (1 - p["prepaid_discount_rate"])
                prepaid_cohorts.append({"start": m, "count": new_pre, "stage":1, "deferred": new_pre * p["monthly_price"] * 9 * (1 - p["prepaid_discount_rate"])})

        # Inventory arrivals
        inv_cost = 0
        for arr in [o for o in pending if o[0] == m]:
            _, s, qty, cost = arr
            inventory[s] += qty
            inv_cost += cost
        pending = [o for o in pending if o[0] > m]

        # Shipments (custom S1 limits)
        ship_mon = {1: 0, 2: 0, 3: 0}
        ship_pre = {1: 0, 2: 0, 3: 0}
        for c in monthly_cohorts:
            age = m - c["start"] + 1
            lim = c["s1_limit"]
            if age <= lim:
                s = 1; c["s1_shipped"] += 1
            elif age <= lim + 3:
                s = 2
            elif age <= lim + 6:
                s = 3
            else:
                continue
            ship_mon[s] += c["count"]
            c["count"] = int(round(c["count"] * (1 - p["churn_rate"])) )
        for c in prepaid_cohorts:
            age = m - c["start"] + 1
            if 1 <= age <= 9:
                s = min(1 + (age - 1) // 3, 3)
                ship_pre[s] += c["count"]

        # Reorder logic
        exp = {s: ship_mon[s] + ship_pre[s] for s in (1,2,3)}
        reorder = []
        reorder_cost = 0
        for s in (1,2,3):
            inventory[s] -= exp[s]
            fut = exp[s] * p["lead_time"]
            thr = math.ceil((exp[s] + fut) * p["reorder_safety"] )
            if inventory[s] <= thr:
                reorder.append(f"S{s}")
                pending.append((m + p["lead_time"], s, p["reorder_qty"], p["reorder_cost"]))
                reorder_cost += p["reorder_cost"]

        # Shipping cost
        ship_cost = sum(exp.values()) * p["shipping_cost_pkg"]

        # Revenue & COGS
        rev_mon = sum(ship_mon.values()) * p["monthly_price"]
        rev_pre = 0
        for c in prepaid_cohorts:
            age = m - c["start"] + 1
            if 1 <= age <= 9:
                slice_rev = c["deferred"] / (10 - age)
                rev_pre += slice_rev
                c["deferred"] -= slice_rev
        cogs_mon = sum(ship_mon.values()) * cost_per_pkg
        cogs_pre = sum(ship_pre.values()) * cost_per_pkg
        total_rev = rev_mon + rev_pre; total_cogs = cogs_mon + cogs_pre
        cac = new_mon * p["cac_new_monthly"] + new_pre * p["cac_new_prepaid"]
        gross = total_rev - total_cogs
        op_inc = gross - cac
        net_income = op_inc - ship_cost
        # cash changes exclude non-cash COGS and recognize monthly revenue
        net = rev_mon - cac - ship_cost - inv_cost
        cash += net

        # Active counts
        act_mon = sum(c["count"] for c in monthly_cohorts if 1 <= (m - c["start"] + 1) <= ((4 - c["stage"]) * 3))
        act_pre = sum(c["count"] for c in prepaid_cohorts if 1 <= (m - c["start"] + 1) <= 9)
        deferred_bal = sum(c["deferred"] for c in prepaid_cohorts)

        records.append({
            "Month": m, "New Monthly Subs": new_mon,
            "New Prepaid Members": new_pre,
            "Stage 1 To Ship": ship_mon[1] + ship_pre[1],
            "Stage 2 To Ship": ship_mon[2] + ship_pre[2],
            "Stage 3 To Ship": ship_mon[3] + ship_pre[3],
            "Inv S1": inventory[1], "Inv S2": inventory[2], "Inv S3": inventory[3],
            "Reorder": ",".join(reorder),
            "Active Monthly Subs": act_mon, "Active Prepaid Members": act_pre,
            "Monthly Revenue": round(rev_mon,2), "Prepaid Revenue Recog": round(rev_pre,2),
            "Total Revenue": round(total_rev,2), "Gross Profit": round(gross,2),
            "Operating Income": round(op_inc,2),
            "Net Income": round(net_income,2),
            "COGS Mon": round(cogs_mon,2), "COGS Pre": round(cogs_pre,2),
            "Total COGS": round(total_cogs,2),
            "CAC": round(cac,2), "Shipping Exp": round(ship_cost,2),
            "Reorder Cost": reorder_cost,
            "Net Cash Flow": round(net,2), "Cash Balance": round(cash,2),
            "Deferred Rev Bal": round(deferred_bal,2)
        })

    return pd.DataFrame(records).set_index("Month")

# ─── Financial Statements ──────────────────────────────────────────────────────
def build_financials(df, p):
    total_pkgs = sum(p["initial_inventory"].values())
    bs = pd.DataFrame({"Cash": df["Cash Balance"]})
    bs["Inventory"] = df[["Inv S1", "Inv S2", "Inv S3"]].sum(axis=1) * p["initial_inventory_cost"] / total_pkgs
    bs["Deferred Rev"] = df["Deferred Rev Bal"]
    bs["Total Curr Assets"] = bs["Cash"] + bs["Inventory"]
    bs["Total Liabilities"] = bs["Deferred Rev"]
    bs["Paid-in Capital"] = p["initial_inventory_cost"]
    bs["Retained Earnings"] = df["Net Income"].cumsum()
    bs["Total Equity"] = bs["Paid-in Capital"] + bs["Retained Earnings"]
    bs["Total L+E"] = bs["Total Liabilities"] + bs["Total Equity"]

    is_df = pd.DataFrame({
        "Revenue": df["Total Revenue"],
        "COGS": df["Total COGS"],
        "Gross Profit": df["Gross Profit"],
        "Op Expenses": df["CAC"] + df["Shipping Exp"],
        "Op Income": df["Operating Income"] - df["Shipping Exp"],
        "Net Income": df["Net Income"]
    })

    cf = pd.DataFrame({
        "Op Cash Flow": df["Net Cash Flow"],
        "Fin Cash Flow": [p["initial_inventory_cost"]] + [0] * (len(df) - 1)
    })

    return bs, is_df, cf

# ─── Run & Display ─────────────────────────────────────────────────────────────
sim_df = run_simulation(params)
bs_df, is_df, cf_df = build_financials(sim_df, params)

fmt_int = "{:,}"
fmt_flt = "{:,.2f}"

st.subheader("Monthly Simulation")
st.dataframe(
    sim_df.style
        .format(fmt_int, subset=sim_df.select_dtypes("int").columns)
        .format(fmt_flt, subset=sim_df.select_dtypes("float").columns)
)

st.subheader("Balance Sheet")
st.dataframe(bs_df.style.format(fmt_flt, subset=bs_df.columns))

st.subheader("Income Statement / P&L")
st.dataframe(is_df.style.format(fmt_flt, subset=is_df.columns))

st.subheader("Cash Flow Statement")
st.dataframe(cf_df.style.format(fmt_flt, subset=cf_df.columns))
