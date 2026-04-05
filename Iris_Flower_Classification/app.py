import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("Iris Flower Prediction App")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Input fields
st.header("Enter Flower Details")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    result = iris.target_names[prediction][0]

    st.success(f"Predicted Flower: {result}")