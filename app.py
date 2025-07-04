import streamlit as st
import pandas as pd
import math

# ─── Define Inputs ─────────────────────────────────────────────────────────────
st.title("BareBump Cash‑Flow Simulator")
st.sidebar.header("Parameters")

# Input controls
monthly_price   = st.sidebar.number_input("Sale Price", 0, 500, 75)
init_subs       = st.sidebar.number_input("Initial Subs", 0, 1000, 250)
init_pre        = st.sidebar.number_input("Initial Prepaid", 0, 1000, 20)
growth          = st.sidebar.number_input("Growth Rate", 0.0, 1.0, 0.10, format="%.2f")
pct_pre         = st.sidebar.number_input("% Prepaid", 0.0, 1.0, 0.20, format="%.2f")
disc_pre        = st.sidebar.number_input("Prepaid Discount", 0.0, 1.0, 0.10, format="%.2f")
cac_mon         = st.sidebar.number_input("CAC Monthly", 0, 500, 20)
churn           = st.sidebar.number_input("Churn Rate", 0.0, 1.0, 0.05, format="%.2f")
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
    "churn_rate":              churn,
    "ship1_dist":              {1: ship1_1, 2: ship1_2, 3: ship1_3},
    "simulation_months":       months
}

# ─── Simulation Function ─────────────────────────────────────────────────────────
def run_simulation(params):
    records = []
    cash_balance = 0
    monthly_cohorts = []

    # Month 1 seed cohort with Stage1 limits
    init_mon = params["initial_subscribers"]
    for limit, pct in params["ship1_dist"].items():
        count = int(round(init_mon * pct))
        if count > 0:
            monthly_cohorts.append({
                "start": 1,
                "count": count,
                "stage": 1,
                "s1_limit": limit,
                "s1_shipped": 0
            })

    for month in range(1, params["simulation_months"] + 1):
        # Determine new subscriptions (only Month1 for this example)
        new_mon = 0
        if month == 1:
            new_mon = init_mon

        # Shipments loop with custom Stage1 limits and churn
        ship_mon = {1: 0, 2: 0, 3: 0}
        for cohort in monthly_cohorts:
            age = month - cohort["start"] + 1
            if age <= 0:
                continue

            # Determine which stage to ship
            if age <= cohort["s1_limit"]:
                stage = 1
                cohort["s1_shipped"] += 1
            elif age <= cohort["s1_limit"] + 3:
                stage = 2
            elif age <= cohort["s1_limit"] + 6:
                stage = 3
            else:
                continue

            # Ship and apply churn
            ship_mon[stage] += cohort["count"]
            cohort["count"] = int(round(cohort["count"] * (1 - params["churn_rate"])) )

        # Calculate revenue, CAC, net flow, update cash
        revenue = sum(ship_mon.values()) * params["monthly_price"]
        cac     = new_mon * params["cac_new_monthly"]
        net     = revenue - cac
        cash_balance += net

        records.append({
            "Month": month,
            "New Monthly Subs": new_mon,
            "Stage 1 To Ship": ship_mon[1],
            "Stage 2 To Ship": ship_mon[2],
            "Stage 3 To Ship": ship_mon[3],
            "Monthly Revenue": revenue,
            "CAC Expense": cac,
            "Net Cash Flow": net,
            "Cash Balance": cash_balance
        })

    return pd.DataFrame(records).set_index("Month")

# Run simulation
df = run_simulation(data)

# ─── Display with comma + 2‑dec formats ─────────────────────────────────────────
st.subheader("Monthly Simulation")
styled = df.style.format(
    {col: ",.2f" for col in df.select_dtypes(include=['float']).columns}
).format(
    {col: ","    for col in df.select_dtypes(include=['int']).columns}
)
st.dataframe(styled)

