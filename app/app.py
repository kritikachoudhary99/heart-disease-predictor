import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("Heart Disease Predictor 🫀")
st.write("Enter Patient Details:")

name = st.text_input("Patient Name")
age = st.slider("Age", 20, 80, 50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (0=None, 1=Mild, 2=Moderate, 3=Severe)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thal", [1, 2, 3])

if st.button("Predict"):
    if name == "":
        st.warning("Please enter patient name!")
    else:
        columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=columns)
        data_scaled = scaler.transform(data)
        result = model.predict(data_scaled)

        if result[0] == 1:
            st.error(f" {name}, Heart Disease Detected!")
        else:
            st.success(f" {name}, No Heart Disease!")