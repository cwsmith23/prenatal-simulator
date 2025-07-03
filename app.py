# Sidebar inputs with slider + number_input for each param
st.sidebar.header("Parameters")

def slider_with_input(label, min_val, max_val, default, step, is_float=False, fmt="%d"):
    col1, col2 = st.sidebar.columns([3,1])
    if is_float:
        val = col1.slider(label, float(min_val), float(max_val), float(default), float(step))
        typed = col2.number_input(
            "", min_value=float(min_val), max_value=float(max_val),
            value=float(val), step=float(step), format=fmt
        )
    else:
        val = col1.slider(label, int(min_val), int(max_val), int(default), int(step))
        typed = col2.number_input(
            "", min_value=int(min_val), max_value=int(max_val),
            value=int(val), step=int(step), format=fmt
        )
    return typed

monthly_price = slider_with_input("Price", 0, 500, 75, 1)
init_subs     = slider_with_input("Initial Subs", 0, 2000, 250, 10)
init_pre      = slider_with_input("Initial Prepaid", 0, 1000, 20, 10)
growth        = slider_with_input("Growth Rate", 0.0, 1.0, 0.10, 0.01, is_float=True, fmt="%.2f")
pct_pre       = slider_with_input("% Prepaid",  0.0, 1.0, 0.20, 0.01, is_float=True, fmt="%.2f")
disc_pre      = slider_with_input("Prepaid Disc",0.0, 1.0, 0.10, 0.01, is_float=True, fmt="%.2f")
churn         = slider_with_input("Churn Rate", 0.0, 1.0, 0.05, 0.01, is_float=True, fmt="%.2f")
lead_time     = slider_with_input("Lead Time", 0, 6, 1, 1)
safety        = slider_with_input("Safety ×", 1.0, 3.0, 1.2, 0.05, is_float=True, fmt="%.2f")
rqty          = slider_with_input("Reorder Qty", 0, 5000, 1330, 10)
rcost         = slider_with_input("Reorder Cost",0,100000,25000,1000)
inv1          = slider_with_input("Inv Stage 1", 0,5000,1330,10)
inv2          = slider_with_input("Inv Stage 2", 0,5000,1330,10)
inv3          = slider_with_input("Inv Stage 3", 0,5000,1330,10)
inv_cost      = slider_with_input("Inv Cost", 0,200000,75000,1000)
st1           = slider_with_input("Start S1", 0.0,1.0,0.60,0.01, is_float=True, fmt="%.2f")
st2           = slider_with_input("Start S2", 0.0,1.0,0.30,0.01, is_float=True, fmt="%.2f")
st3           = slider_with_input("Start S3", 0.0,1.0,0.10,0.01, is_float=True, fmt="%.2f")
months        = slider_with_input("Months",1,36,12,1)
