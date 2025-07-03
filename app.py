import streamlit as st
import pandas as pd
import math

def run_simulation(params):
    months = params["simulation_months"]
    total_pkgs = sum(params["initial_inventory"].values())
    cost_per_pkg = params["initial_inventory_cost"] / total_pkgs

    inventory = {s:int(q) for s,q in params["initial_inventory"].items()}
    cash_balance = -int(params["initial_inventory_cost"])
    pending_orders = []
    monthly_cohorts = []
    prepaid_cohorts = []

    # Month 1 seeds
    init_pre = params["initial_prepaid"]
    init_mon = params["initial_subscribers"]
    if init_mon>0:
        monthly_cohorts.append({"start":1,"count":init_mon,"stage":1})
    if init_pre>0:
        prepaid_cohorts.append({"start":1,"count":init_pre,"stage":1})

    records = []
    for month in range(1, months+1):
        if month==1:
            new_mon, new_pre = init_mon, init_pre
        else:
            alive = sum(c["count"] for c in monthly_cohorts+prepaid_cohorts)
            new_tot = alive * params["subscriber_growth_rate"]
            new_pre = int(round(new_tot * params["percent_prepaid"]))
            new_mon = int(round(new_tot - new_pre))
            # distribute by starting stage
            for st,pct in params["start_stage_dist"].items():
                cnt = int(round(new_mon * pct))
                if cnt>0:
                    monthly_cohorts.append({"start":month,"count":cnt,"stage":st})
            if new_pre>0:
                prepaid_cohorts.append({"start":month,"count":new_pre,"stage":1})

        prepaid_rev = new_pre * params["monthly_price"] * 9 * (1-params["prepaid_discount_rate"])

        # arrive
        arrivals = [o for o in pending_orders if o[0]==month]
        for _,st,qty in arrivals:
            inventory[st]+=qty
        pending_orders = [o for o in pending_orders if o[0]>month]

        # ship
        ship_mon = {1:0,2:0,3:0}
        ship_pre = {1:0,2:0,3:0}
        for c in monthly_cohorts:
            age = month - c["start"] + 1
            max_age = (4-c["stage"])*3
            if 1<=age<=max_age:
                s = min(c["stage"]+(age-1)//3,3)
                ship_mon[s]+=c["count"]
                c["count"] = int(round(c["count"]*(1-params["churn_rate"])))
        for c in prepaid_cohorts:
            age = month - c["start"] + 1
            if 1<=age<=9:
                s = min(1+(age-1)//3,3)
                ship_pre[s]+=c["count"]

        # reorder
        expected = {s:ship_mon[s]+ship_pre[s] for s in (1,2,3)}
        inv_cost=0; reorder=[]
        for s in (1,2,3):
            inventory[s]-=expected[s]
            thr = math.ceil(expected[s]*params["reorder_safety"])
            if inventory[s]<=thr:
                pending_orders.append((month+params["lead_time"],s,params["reorder_qty"]))
                inv_cost+=params["reorder_cost"]
                reorder.append(s)

        # financials
        rev_mon = sum(ship_mon.values())*params["monthly_price"]
        total_rev = rev_mon + prepaid_rev
        cac = new_mon*params["cac_new_monthly"] + new_pre*params["cac_new_prepaid"]
        cogs = sum(expected.values())*cost_per_pkg
        net = total_rev - cac - cogs - inv_cost
        cash_balance += net

        s1 = ship_mon[1]+ship_pre[1]
        s2 = ship_mon[2]+ship_pre[2]
        s3 = ship_mon[3]+ship_pre[3]

        records.append({
            "Month": month,
            "New Monthly Subs": new_mon,
            "New Prepaid Subs": new_pre,
            "Stage 1 Subs": s1,
            "Stage 2 Subs": s2,
            "Stage 3 Subs": s3,
            "Monthly Revenue": round(rev_mon,2),
            "Prepaid Revenue": round(prepaid_rev,2),
            "Total Revenue": round(total_rev,2),
            "CAC Expense": round(cac,2),
            "COGS": round(cogs,2),
            "Inv S1": inventory[1],
            "Inv S2": inventory[2],
            "Inv S3": inventory[3],
            "Reorder": reorder,
            "Net Cash Flow": round(net,2),
            "Cash Balance": round(cash_balance,2)
        })

    return pd.DataFrame(records)

st.title("Prenatal Subscription Simulator")

# Sidebar inputs
st.sidebar.header("Parameters")
monthly_price = st.sidebar.slider("Price", 0, 500, 75)
init_subs     = st.sidebar.slider("Initial Subs", 0, 2000, 250)
init_pre      = st.sidebar.slider("Initial Prepaid", 0, 1000, 20)
growth        = st.sidebar.slider("Growth Rate", 0.0, 1.0, 0.10)
pct_pre       = st.sidebar.slider("% Prepaid",  0.0, 1.0, 0.20)
disc_pre      = st.sidebar.slider("Prepaid Disc",0.0, 1.0, 0.10)
churn         = st.sidebar.slider("Churn Rate", 0.0, 1.0, 0.05)
lead_time     = st.sidebar.slider("Lead Time", 0, 6, 1)
safety        = st.sidebar.slider("Safety ×", 1.0, 3.0, 1.2)
rqty          = st.sidebar.slider("Reorder Qty", 0, 5000, 1330)
rcost         = st.sidebar.slider("Reorder Cost",0,100000,25000)
inv1          = st.sidebar.slider("Inv Stage 1", 0,5000,1330)
inv2          = st.sidebar.slider("Inv Stage 2", 0,5000,1330)
inv3          = st.sidebar.slider("Inv Stage 3", 0,5000,1330)
inv_cost      = st.sidebar.number_input("Inv Cost", value=75000)
st1           = st.sidebar.slider("Start S1", 0.0,1.0,0.60)
st2           = st.sidebar.slider("Start S2", 0.0,1.0,0.30)
st3           = st.sidebar.slider("Start S3", 0.0,1.0,0.10)
months        = st.sidebar.slider("Months",1,36,12)

params = {
    "monthly_price": monthly_price,
    "initial_subscribers": init_subs,
    "initial_prepaid": init_pre,
    "subscriber_growth_rate": growth,
    "percent_prepaid": pct_pre,
    "prepaid_discount_rate": disc_pre,
    "cac_new_monthly": 20,
    "cac_new_prepaid": 20,
    "initial_inventory": {1:inv1,2:inv2,3:inv3},
    "initial_inventory_cost": inv_cost,
    "reorder_qty": rqty,
    "reorder_cost": rcost,
    "churn_rate": churn,
    "lead_time": lead_time,
    "reorder_safety": safety,
    "start_stage_dist": {1:st1,2:st2,3:st3},
    "simulation_months": months
}

df = run_simulation(params)
st.dataframe(df)
