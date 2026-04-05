import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("💳 Credit Card Fraud Detection App")

# Load dataset
df = pd.read_csv("creditcard.csv")

# -----------------------
# Preprocessing
# -----------------------

# Features & Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# -----------------------
# User Input
# -----------------------

st.header("Enter Transaction Details")

# Important features
time = st.number_input("Transaction Time", min_value=0.0, value=10000.0)
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

# Some V features (simplified input)
v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)

# -----------------------
# Prediction
# -----------------------

if st.button("Check Transaction"):

    # Create full input (30 features)
    input_data = [time]

    # V1 to V28 (fill missing with 0)
    v_features = [v1, v2, v3] + [0]*25

    input_data.extend(v_features)
    input_data.append(amount)

    input_array = np.array([input_data])

    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")