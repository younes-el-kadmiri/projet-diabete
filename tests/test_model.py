import joblib
import pandas as pd

model = joblib.load("model/model.pkl")

def test_prediction():
    sample = pd.DataFrame([{
        "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66,
        "SkinThickness": 29, "Insulin": 0, "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351, "Age": 31
    }])
    pred = model.predict(sample)
    assert pred in [0, 1], "La prédiction doit être binaire"

test_prediction()
