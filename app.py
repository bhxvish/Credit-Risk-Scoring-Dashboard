import streamlit as st
import pandas as pd
import joblib
from explain import get_shap_values
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Credit Risk Scoring Dashboard")

model = joblib.load("model.pkl")

st.sidebar.header("Enter Applicant Info")
input_data = {
    "age": st.sidebar.slider("Age", 21, 65, 30),
    "income": st.sidebar.number_input("Annual Income", value=50000),
    "loan_amount": st.sidebar.number_input("Loan Amount", value=15000),
    "loan_term_months": st.sidebar.slider("Loan Term (months)", 12, 60, 36),
    "credit_score": st.sidebar.slider("Credit Score", 300, 850, 600),
    "num_credit_cards": st.sidebar.slider("Credit Cards", 1, 10, 2)
}

input_df = pd.DataFrame([input_data])

pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")
st.write(f"**Risk Level:** {'High' if pred==1 else 'Low'}")
st.write(f"**Default Probability:** {prob:.2%}")

st.subheader("SHAP Explanation")
shap_values = get_shap_values(input_df)

fig, ax = plt.subplots()
shap_values = get_shap_values(input_df)
shap.plots.waterfall(shap_values[0], max_display=6, show=False)

st.pyplot(fig)
