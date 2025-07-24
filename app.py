import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("xgboost_diabetes_model.pkl")

st.title("🌡️ Diabetes Risk Predictor")

# Mapping
yes_no = {"Yes": 1, "No": 0}

# Input Form
def user_input():
    col1, col2 = st.columns(2)

    with col1:
        CholCheck = st.selectbox("Had Cholesterol Check", ["No", "Yes"])
        BMI = st.number_input("BMI", 10.0, 60.0, 25.0)
        Smoker = st.selectbox("Do you Smoke?", ["No", "Yes"])
        Stroke = st.selectbox("Ever had Stroke?", ["No", "Yes"])
        HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack?", ["No", "Yes"])
        PhysActivity = st.selectbox("Physically Active?", ["No", "Yes"])
        HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"])
        GenHlth = st.slider("General Health (1 = Excellent, 5 = Poor)", 1, 5, 3)
        MentHlth = st.number_input("Poor Mental Health Days (last 30)", 0, 30, 0)
        PhysHlth = st.number_input("Poor Physical Health Days (last 30)", 0, 30, 0)
        DiffWalk = st.selectbox("Difficulty Walking?", ["No", "Yes"])

    with col2:
        HighBP = st.selectbox("High Blood Pressure", ["No", "Yes"])
        HighChol = st.selectbox("High Cholesterol", ["No", "Yes"])
        age_mapping = {
            "18–24": 1, "25–29": 2, "30–34": 3, "35–39": 4,
            "40–44": 5, "45–49": 6, "50–54": 7, "55–59": 8,
            "60–64": 9, "65–69": 10, "70–74": 11, "75–79": 12, "80+": 13
        }
        age_label = st.selectbox("Age Category", list(age_mapping.keys()))
        Age = age_mapping[age_label]


        education_mapping = {
            "Never Attended": 1,
            "Grades 1–8": 2,
            "Grades 9–11": 3,
            "High School Grad": 4,
            "Some College": 5,
            "College Grad": 6
        }
        education_label = st.selectbox("Education Level", list(education_mapping.keys()))
        Education = education_mapping[education_label]

        income_mapping = {
            "<$10K": 1,
            "$10K–$15K": 2,
            "$15K–$20K": 3,
            "$20K–$25K": 4,
            "$25K–$35K": 5,
            "$35K–$50K": 6,
            "$50K–$75K": 7,
            "$75K+": 8
        }
        income_label = st.selectbox("Income Range", list(income_mapping.keys()))
        Income = income_mapping[income_label]


        Has_High_Risk_Factor = st.selectbox("High Risk Factor Present?", ["No", "Yes"])
        Cardio_History = st.selectbox("Cardio History Present?", ["No", "Yes"])
        GenHlth_Binary = st.selectbox("General Health (Binary - Good/Bad)", ["Good", "Bad"])

        st.markdown("#### Healthy Lifestyle Habits")
        exercise = st.checkbox("Regular Physical Activity")
        fruits = st.checkbox("Eats Fruits Regularly")
        veggies = st.checkbox("Eats Vegetables Regularly")
        not_smoking = Smoker == "No"
        healthy_score = sum([exercise, fruits, veggies, not_smoking])

    # Compute derived fields
    Total_Unhealthy_Days = int(MentHlth) + int(PhysHlth)
    BMI_Category_Obese = 1 if BMI >= 30 else 0
    BMI_Category_Overweight = 1 if 25 <= BMI < 30 else 0
    Age_Group_30_44 = 1 if 3 <= Age <= 5 else 0
    Age_Group_60_plus = 1 if Age >= 9 else 0

    # Build input DataFrame
    data = pd.DataFrame([[
        yes_no[HighBP], yes_no[HighChol], yes_no[CholCheck], BMI,
        yes_no[Smoker], yes_no[Stroke], yes_no[HeartDiseaseorAttack],
        yes_no[PhysActivity], yes_no[HvyAlcoholConsump], GenHlth,
        MentHlth, PhysHlth, yes_no[DiffWalk], Age, Education, Income,
        yes_no[Has_High_Risk_Factor], yes_no[Cardio_History],
        healthy_score, Total_Unhealthy_Days,
        0 if GenHlth_Binary == "Good" else 1,
        BMI_Category_Obese, BMI_Category_Overweight,
        Age_Group_30_44, Age_Group_60_plus
    ]], columns=[
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Education', 'Income',
        'Has_High_Risk_Factor', 'Cardio_History', 'Healthy_Lifestyle_Score',
        'Total_Unhealthy_Days', 'GenHlth_Binary', 'BMI_Category_Obese',
        'BMI_Category_Overweight', 'Age_Group_30-44', 'Age_Group_60+'
    ])

    return data

# Run App
input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes ({prob * 100:.2f}% confidence)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({(1 - prob) * 100:.2f}% confidence)")
