import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'diabetes_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Set a custom page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Header Section with Styling
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #2E86C1;">🩺 Diabetes Prediction App</h1>
        <p style="font-size: 18px; color: #566573;">Effortlessly Predict Diabetes Using Medical Parameters</p>
    </div>
    """, 
    unsafe_allow_html=True
)

st.markdown("---")

# Sidebar for Input
st.sidebar.title("📝 Enter Patient Details")
st.sidebar.markdown("Fill in the medical details below to make a prediction:")

# Input fields in sidebar
with st.sidebar.form("input_form"):
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, step=1)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1)
    insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)
    submitted = st.form_submit_button("🔍 Predict")

# Preprocessing function to match training data format
def preprocess_input(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    input_df = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree_function],
        "Age": [age],
    })

    # Add one-hot encoded categorical variables with default values
    input_df["NewBMI_Obesity 1"] = 0
    input_df["NewBMI_Obesity 2"] = 0
    input_df["NewBMI_Obesity 3"] = 0
    input_df["NewBMI_Overweight"] = 0
    input_df["NewBMI_Underweight"] = 0
    input_df["NewInsulinScore_Normal"] = 0
    input_df["NewGlucose_Low"] = 0
    input_df["NewGlucose_Normal"] = 0
    input_df["NewGlucose_Overweight"] = 0
    input_df["NewGlucose_Secret"] = 0

    feature_order = model.feature_names_in_
    input_df = input_df[feature_order]
    return input_df

# Main Section
if submitted:
    st.markdown("---")
    st.header("🔍 Prediction Results")

    input_data = preprocess_input(
        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age
    )

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display the result in columns
    col1, col2 = st.columns(2)

    with col1:
        if prediction[0] == 1:
            st.error("🚨 **Result:** The patient is likely to have diabetes.")
        else:
            st.success("✅ **Result:** The patient is unlikely to have diabetes.")

    with col2:
        st.subheader("📊 Prediction Probability:")
        prob_df = pd.DataFrame(prediction_proba, columns=["Non-Diabetic", "Diabetic"])
        st.bar_chart(prob_df.T)

    # Summary Card
    st.markdown(
        """
        <div style="background-color: #FAD7A0; padding: 10px; border-radius: 10px;">
            <h3>🔔 Model Highlights:</h3>
            <ul>
                <li><strong>Non-Diabetic Probability:</strong> {:.2f}%</li>
                <li><strong>Diabetic Probability:</strong> {:.2f}%</li>
            </ul>
        </div>
        """.format(prediction_proba[0][0] * 100, prediction_proba[0][1] * 100),
        unsafe_allow_html=True
    )

    st.markdown("---")

# Footer
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px;">
        <p style="color: #566573;">Developed with ❤️ by Saloni Bansal</p>
    </div>
    """,
    unsafe_allow_html=True
)
