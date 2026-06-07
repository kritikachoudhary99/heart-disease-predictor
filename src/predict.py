# Heart Disease Predictor - Save Model & Predict
import joblib
from preprocess import load_and_preprocess
from train import train_models

# Data load karo
X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

# Models train karo
model = train_models(X_train, X_test, y_train, y_test)

# Model save karo
joblib.dump(model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("\nModel saved!")

# Ek naye patient ka prediction karo
# [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
import numpy as np
new_patient = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)

if prediction[0] == 1:
    print("\nResult: Heart Disease DETECTED!")
else:
    print("\nResult: No Heart Disease!")