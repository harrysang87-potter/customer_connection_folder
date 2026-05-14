import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline (cached so it loads only once)
@st.cache_resource
def load_model():
    return joblib.load("kplc_bill_model.pkl")

model = load_model()

# ── Page layout ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="KPLC Bill Estimator", layout="centered")
st.title("KPLC Bill Estimator")
st.write("Enter the customer details to estimate the monthly bill before it is generated.")
st.divider()

# ── Input widgets ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    units = st.number_input("Units consumed (kWh)",
                            min_value=0, max_value=100000, value=500, step=50)
    month = st.selectbox("Billing month", options=list(range(1, 13)),
                         format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
    cust_type = st.selectbox("Customer type",
                             ["Domestic","Commercial","Industrial","Agricultural"])

with col2:
    region = st.selectbox("Region",
                          ["Nairobi","Mombasa","Kisumu","Nakuru","Eldoret",
                           "Thika","Nyeri","Garissa","Meru","Kisii"])
    credit = st.slider("Credit score", min_value=0, max_value=100, value=70)
    year   = st.selectbox("Billing year", [2023, 2024])

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Estimate Bill", type="primary"):

    is_peak = 1 if month in [1, 2, 7, 8, 12] else 0

    input_data = pd.DataFrame({
        "units_consumed": [units],
        "credit_score":   [credit],
        "billing_month":  [month],
        "billing_year":   [year],
        "is_peak_month":  [is_peak],
        "customer_type":  [cust_type],
        "region":         [region],
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated bill: KES {prediction:,.0f}")
    st.caption(f"Peak month flag: {'Yes (Jan/Feb/Jul/Aug/Dec)' if is_peak else 'No'}")
    st.caption(f"Credit score used: {credit}  |  Region: {region}  |  Type: {cust_type}")
