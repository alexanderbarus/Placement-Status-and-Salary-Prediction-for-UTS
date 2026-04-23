from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
# Load the machine learning model
pipeline_cls = joblib.load('placement_status_cls.pkl')
pipeline_reg = joblib.load('salary_reg.pkl')

class placementStatusFeatures(BaseModel):
    gender: str
    ssc_percentage: int
    hsc_percentage: int
    degree_percentage: int
    cgpa: float
    technical_skill_score: int
    soft_skill_score: int
    certifications: int
    extracurricular_activities: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict")
def predict(payload: placementStatusFeatures):

    try:
        # convert to dataframe
        input_df = pd.DataFrame([payload.dict()])
        
        # predictions
        prediction_cls = pipeline_cls.predict(input_df)[0]
        prediction_reg = pipeline_reg.predict(input_df)[0]

        return {
            "prediction_classifier": int(prediction_cls),
            "prediction_regression": float(prediction_reg)
        }

    except Exception as e:
        return {
            "error": str(e)
        }