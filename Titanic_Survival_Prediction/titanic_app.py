import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Title

st.title("🚢 Titanic Survival Prediction App")

# Load dataset

df = pd.read_csv("Titanic-Dataset.csv")

# -----------------------

# Preprocessing

# -----------------------

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df['Embarked'].fillna('S', inplace=True)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Features & Target

X = df.drop('Survived', axis=1)
y = df['Survived']

# Train model

model = RandomForestClassifier()
model.fit(X, y)

# -----------------------

# User Input

# -----------------------

st.header("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0)
fare = st.number_input("Fare", min_value=0.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Convert inputs

sex = 0 if sex == "male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

# -----------------------

# Prediction

# -----------------------

if st.button("Predict"):
    input_data = np.array([[0, pclass, sex, age, sibsp, parch, fare, embarked]])


    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")

