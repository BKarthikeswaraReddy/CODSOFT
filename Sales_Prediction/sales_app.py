import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("📊 Sales Prediction App")

# Load dataset
df = pd.read_csv("advertising.csv")

# Features & Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# -----------------------
# User Input
# -----------------------

st.header("Enter Advertising Budget")

tv = st.number_input("TV Budget", min_value=0.0)
radio = st.number_input("Radio Budget", min_value=0.0)
newspaper = st.number_input("Newspaper Budget", min_value=0.0)

# -----------------------
# Prediction
# -----------------------

if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)

    st.success(f"📈 Predicted Sales: {prediction[0]:.2f}")