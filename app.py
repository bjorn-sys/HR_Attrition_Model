import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
with open("xgb_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoders.pkl", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

st.set_page_config(page_title="HR Attrition Predictor", layout="centered")
st.title("🧠 HR Attrition Prediction App")
st.markdown("This app predicts whether an employee is likely to leave the company based on HR data.")

# Sidebar: Show categorical encoding info
with st.sidebar.expander("ℹ️ Categorical Encoding Info"):
    st.markdown("**Label Encodings for Categorical Features:**")
    for col, le in label_encoders.items():
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        mapping_str = "\n".join([f"- {k} → {v}" for k, v in mapping.items()])
        st.markdown(f"**{col}**\n{mapping_str}")

# Input form
def get_user_input():
    education_options = {
        1: "1 - No Formal Education",
        2: "2 - High School / Secondary",
        3: "3 - Associate / Diploma",
        4: "4 - Bachelor's Degree",
        5: "5 - Graduate Degree or Higher"
    }

    input_dict = {
        'Age': st.slider("Age", 18, 60, 30),
        'BusinessTravel': st.selectbox("Business Travel", label_encoders['BusinessTravel'].classes_),
        'DailyRate': st.slider("Daily Rate", 100, 1500, 800),
        'Department': st.selectbox("Department", label_encoders['Department'].classes_),
        'DistanceFromHome': st.slider("Distance From Home (km)", 0, 30, 5),
        'Education': st.selectbox("Education Level", options=list(education_options.keys()), format_func=lambda x: education_options[x]),
        'EducationField': st.selectbox("Education Field", label_encoders['EducationField'].classes_),
        'EnvironmentSatisfaction': st.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4]),
        'Gender': st.selectbox("Gender", label_encoders['Gender'].classes_),
        'HourlyRate': st.slider("Hourly Rate", 30, 150, 60),
        'JobInvolvement': st.selectbox("Job Involvement (1-4)", [1, 2, 3, 4]),
        'JobLevel': st.selectbox("Job Level", [1, 2, 3, 4, 5]),
        'JobRole': st.selectbox("Job Role", label_encoders['JobRole'].classes_),
        'JobSatisfaction': st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4]),
        'MaritalStatus': st.selectbox("Marital Status", label_encoders['MaritalStatus'].classes_),
        'MonthlyIncome': st.slider("Monthly Income", 1000, 20000, 5000),
        'MonthlyRate': st.slider("Monthly Rate", 2000, 27000, 10000),
        'NumCompaniesWorked': st.slider("Number of Companies Worked", 0, 10, 2),
        'OverTime': st.selectbox("OverTime", label_encoders['OverTime'].classes_),
        'PercentSalaryHike': st.slider("Percent Salary Hike", 10, 25, 15),
        'PerformanceRating': st.selectbox("Performance Rating", [1, 2, 3, 4]),
        'RelationshipSatisfaction': st.selectbox("Relationship Satisfaction (1-4)", [1, 2, 3, 4]),
        'StockOptionLevel': st.selectbox("Stock Option Level", [0, 1, 2, 3]),
        'TotalWorkingYears': st.slider("Total Working Years", 0, 40, 10),
        'TrainingTimesLastYear': st.slider("Trainings Last Year", 0, 6, 2),
        'WorkLifeBalance': st.selectbox("Work Life Balance (1-4)", [1, 2, 3, 4]),
        'YearsAtCompany': st.slider("Years at Company", 0, 30, 5),
        'YearsInCurrentRole': st.slider("Years in Current Role", 0, 20, 4),
        'YearsSinceLastPromotion': st.slider("Years Since Last Promotion", 0, 15, 2),
        'YearsWithCurrManager': st.slider("Years with Current Manager", 0, 20, 3),
    }

    return pd.DataFrame([input_dict])

# Collect user input
input_df = get_user_input()

# Encode categorical inputs
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Predict
if st.button("🔍 Predict Attrition"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    attrition_label = label_encoders['Attrition'].inverse_transform([pred])[0]

    st.subheader(f"Prediction: **{attrition_label}**")
    st.metric("Probability of Leaving", f"{prob:.2%}")

    if attrition_label == "Yes":
        st.warning("⚠️ This employee is likely to leave. Consider retention strategies.")
    else:
        st.success("✅ This employee is likely to stay.")
