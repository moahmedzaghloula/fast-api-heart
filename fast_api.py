from fastapi import FastAPI, Form
import numpy as np
import pandas as pd

app = FastAPI()

# Load the model
model = pd.read_pickle("heart.pkl")

@app.get('/')
async def index():
    return {"message": "Welcome to the Heart API!"}

@app.post("/predict")
async def predict(age: int = Form(...), max_heart_rate: int = Form(...), ecg: str = Form(...),
                  st_slope: str = Form(...), blood_pressure: int = Form(...), old_peak: float = Form(...),
                  chest_pain_type: str = Form(...), exercise_angina: str = Form(...),
                  cholesterol: int = Form(...), gender: str = Form(...),
                  fasting_blood_sugar: str = Form(...)):
    patient_fasting_blood_sugar = 1 if fasting_blood_sugar == "Less Than 120 mg/dl" else 0

    new_data = [age, blood_pressure, cholesterol,
                patient_fasting_blood_sugar, max_heart_rate, old_peak]

    patient_gender = [1] if gender == "Male" else [0]

    patient_chest_pain_type = [0, 0, 0]
    if chest_pain_type == "Typical Angina":
        patient_chest_pain_type = [0, 0, 1]
    elif chest_pain_type == "Atypical Angina":
        patient_chest_pain_type = [1, 0, 0]
    elif chest_pain_type == "Non-anginal Pain":
        patient_chest_pain_type = [0, 1, 0]

    patinet_ecg = [1, 0] if ecg == "Normal" else [0, 1] if ecg == "ST" else [0, 0, 1]

    patient_exercise_angina = [1] if exercise_angina == "Yes" else [0]

    patient_slope = [1, 0] if st_slope == "Flat" else [0, 1] if st_slope == "Up" else [0, 0, 1]

    new_data.extend(patient_gender)
    new_data.extend(patient_chest_pain_type)
    new_data.extend(patinet_ecg)
    new_data.extend(patient_exercise_angina)
    new_data.extend(patient_slope)

    predicted_value = model.predict([new_data])[0]
    predicted_label = "Heart Patient" if predicted_value == 1 else "Not Heart Patient"


    return {
        "predicted_label": predicted_label,
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=5012)