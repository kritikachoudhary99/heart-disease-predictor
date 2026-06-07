import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

print("=== Heart Disease Predictor ===")
name = input("Patient Name: ")
age = int(input("Age (20-80): "))
sex = int(input("Sex (0=Female, 1=Male): "))
cp = int(input("Chest Pain Type (0-3): "))
trestbps = int(input("Resting Blood Pressure (90-200): "))
chol = int(input("Cholesterol (100-600): "))
fbs = int(input("Fasting Blood Sugar >120? (0=No, 1=Yes): "))
restecg = int(input("Resting ECG (0-2): "))
thalach = int(input("Max Heart Rate (70-210): "))
exang = int(input("Exercise Angina? (0=No, 1=Yes): "))
oldpeak = float(input("Oldpeak (0.0-6.0): "))
slope = int(input("Slope (0-2): "))
ca = int(input("Number of Vessels (0-3): "))
thal = int(input("Thal (1-3): "))

columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=columns)
data_scaled = scaler.transform(data)
result = model.predict(data_scaled)

if result[0] == 1:
    print(f"\n  {name}, Heart Disease Detected!")
else:
    print(f"\n  {name}, No Heart Disease!")