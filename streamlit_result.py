import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests

def user_input():
    st.sidebar.header("Student Profile Input")

    gender = st.sidebar.radio("Gender", ["Male", "Female"])

    ssc_percentage = st.sidebar.slider("SSC Percentage", 0, 100, 50)
    hsc_percentage = st.sidebar.slider("HSC Percentage", 0, 100, 50)
    degree_percentage = st.sidebar.slider("Degree Percentage", 0, 100, 55)
    cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 5.0)

    technical_skill_score = st.sidebar.slider("Technical Skill Score", 0, 100, 40)
    soft_skill_score = st.sidebar.slider("Soft Skill Score", 0, 100, 40)

    certifications = st.sidebar.number_input("Certifications", 0, 20, 0)

    extracurricular_activities = st.sidebar.radio(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    return {
        "gender": gender,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "cgpa": cgpa,
        "technical_skill_score": technical_skill_score,
        "soft_skill_score": soft_skill_score,
        "certifications": certifications,
        "extracurricular_activities": extracurricular_activities
    }
    
    
def make_prediction(features):
    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=features
        )
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def main():
    st.title("Placement Prediction System")
    
    features = user_input()
    
    result = None
    
    st.markdown("---")
    
    st.subheader("Prediction")
    
    if st.button("Make Prediction"):
        result = make_prediction(features)
    
    if result:
        st.write("API Response:", result)

        classifier = result.get("prediction_classifier")
        regression = result.get("prediction_regression")

        st.write("### Classification Result:", classifier)
        st.write("### Expected Salary (LPA):", regression)
    else:
        st.error("No result returned")
        
if __name__ == '__main__':
    main()