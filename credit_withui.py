import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the saved model
model = joblib.load('C:/Users/Asus/Desktop/research/credit/credit_default_model.pkl')
scaler = joblib.load('C:/Users/Asus/Desktop/research/credit/scaler.pkl')

# Input fields for the 23 variables
st.subheader("Demographic Information")
x1 = st.number_input("X1: Amount of Given Credit (NT dollar)", min_value=0)
x2 = st.selectbox("X2: Gender", options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
x3 = st.selectbox("X3: Education", options=[1, 2, 3, 4], format_func=lambda x: {1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others'}.get(x))
x4 = st.selectbox("X4: Marital Status", options=[1, 2, 3], format_func=lambda x: {1: 'Married', 2: 'Single', 3: 'Others'}.get(x))
x5 = st.slider("X5: Age", 18, 100)

st.subheader("History of Past Payment (1 = delay, -1 = pay duly)")
x6 = st.slider("X6: Repayment Status in September 2005", -1, 9)
x7 = st.slider("X7: Repayment Status in August 2005", -1, 9)
x8 = st.slider("X8: Repayment Status in July 2005", -1, 9)
x9 = st.slider("X9: Repayment Status in June 2005", -1, 9)
x10 = st.slider("X10: Repayment Status in May 2005", -1, 9)
x11 = st.slider("X11: Repayment Status in April 2005", -1, 9)

st.subheader("Amount of Bill Statement (NT dollar)")
x12 = st.number_input("X12: Bill Statement in September 2005", min_value=0)
x13 = st.number_input("X13: Bill Statement in August 2005", min_value=0)
x14 = st.number_input("X14: Bill Statement in July 2005", min_value=0)
x15 = st.number_input("X15: Bill Statement in June 2005", min_value=0)
x16 = st.number_input("X16: Bill Statement in May 2005", min_value=0)
x17 = st.number_input("X17: Bill Statement in April 2005", min_value=0)

st.subheader("Amount of Previous Payment (NT dollar)")
x18 = st.number_input("X18: Amount Paid in September 2005", min_value=0)
x19 = st.number_input("X19: Amount Paid in August 2005", min_value=0)
x20 = st.number_input("X20: Amount Paid in July 2005", min_value=0)
x21 = st.number_input("X21: Amount Paid in June 2005", min_value=0)
x22 = st.number_input("X22: Amount Paid in May 2005", min_value=0)
x23 = st.number_input("X23: Amount Paid in April 2005", min_value=0)

# Create DataFrame from user input
input_data = pd.DataFrame({
    'X1': [x1],
    'X2': [x2],
    'X3': [x3],
    'X4': [x4],
    'X5': [x5],
    'X6': [x6],
    'X7': [x7],
    'X8': [x8],
    'X9': [x9],
    'X10': [x10],
    'X11': [x11],
    'X12': [x12],
    'X13': [x13],
    'X14': [x14],
    'X15': [x15],
    'X16': [x16],
    'X17': [x17],
    'X18': [x18],
    'X19': [x19],
    'X20': [x20],
    'X21': [x21],
    'X22': [x22],
    'X23': [x23]
})

# Feature mapping for model input
feature_mapping = {
    'X1': 'LIMIT_BAL',
    'X2': 'SEX',
    'X3': 'EDUCATION',
    'X4': 'MARRIAGE',
    'X5': 'AGE',
    'X6': 'PAY_0',
    'X7': 'PAY_2',
    'X8': 'PAY_3',
    'X9': 'PAY_4',
    'X10': 'PAY_5',
    'X11': 'PAY_6',
    'X12': 'BILL_AMT1',
    'X13': 'BILL_AMT2',
    'X14': 'BILL_AMT3',
    'X15': 'BILL_AMT4',
    'X16': 'BILL_AMT5',
    'X17': 'BILL_AMT6',
    'X18': 'PAY_AMT1',
    'X19': 'PAY_AMT2',
    'X20': 'PAY_AMT3',
    'X21': 'PAY_AMT4',
    'X22': 'PAY_AMT5',
    'X23': 'PAY_AMT6'
}

# Rename columns to match model features
input_data = input_data.rename(columns=feature_mapping)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Debugging: Show the scaled input
st.write("Scaled Input Data:", input_scaled)

# Make predictions
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

# Debugging: Show the raw prediction output
st.write("Raw Prediction Output:", prediction)
st.write("Raw Probability Output:", probability)

# Display results
if prediction[0] == 1:
    st.write("Prediction: **High risk of default**")
else:
    st.write("Prediction: **Low risk of default**")

# Ensure probability output is formatted correctly
prob_default = probability[0][1]  # Probability of default
st.write(f"Probability of default: **{prob_default* 100:.2f} (or {prob_default * 100:.2f}%)**")