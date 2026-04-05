import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("🎬 Movie Rating Prediction App")

# Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding='latin1')

# -----------------------
# Preprocessing
# -----------------------

# Clean Year
df['Year'] = df['Year'].astype(str).str.replace('-', '')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Clean Duration
df['Duration'] = df['Duration'].astype(str).str.replace(' min', '')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# 🔥 IMPORTANT FIX (Votes cleaning)
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# Handle missing values
df['Rating'].fillna(df['Rating'].median(), inplace=True)
df['Votes'].fillna(df['Votes'].median(), inplace=True)
df['Year'].fillna(df['Year'].median(), inplace=True)
df['Duration'].fillna(df['Duration'].median(), inplace=True)

# Simplify Genre (take first genre only)
df['Genre'] = df['Genre'].astype(str).apply(lambda x: x.split(',')[0])

# Encode Genre
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])

# Features & Target
X = df[['Year', 'Duration', 'Votes', 'Genre']]
y = df['Rating']

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# -----------------------
# User Input
# -----------------------

st.header("Enter Movie Details")

year = st.number_input("Year", min_value=1900, max_value=2025, value=2020)
duration = st.number_input("Duration (minutes)", min_value=1, value=120)
votes = st.number_input("Votes", min_value=0, value=1000)

# Genre dropdown
genre_options = list(le.classes_)
genre_input = st.selectbox("Genre", genre_options)

# Convert genre input
genre_encoded = le.transform([genre_input])[0]

# -----------------------
# Prediction
# -----------------------

if st.button("Predict Rating"):
    input_data = np.array([[year, duration, votes, genre_encoded]])
    prediction = model.predict(input_data)

    st.success(f"⭐ Predicted Rating: {prediction[0]:.2f}")