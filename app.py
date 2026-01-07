import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/covid_prediction_model.pkl')

st.set_page_config(page_title="COVID-19 Prediction System")
st.title("🦠 COVID-19 Clinical Risk Prediction")
st.write("Machine Learning model for early infection detection.")

# Input Form
with st.container():
    st.subheader("Patient Clinical Data")
    age = st.number_input("Patient Age", 1, 100, 25)
    fever = st.selectbox("Fever", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    cough = st.selectbox("Dry Cough", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    breath = st.selectbox("Shortness of Breath", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    throat = st.selectbox("Sore Throat", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

if st.button("Predict Infection Risk"):
    features = pd.DataFrame([[fever, cough, breath, throat, age]], 
                            columns=['fever', 'dry_cough', 'shortness_of_breath', 'sore_throat', 'age'])
    
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    
    if prediction[0] == 1:
        st.error(f"High Risk of COVID-19. Confidence: {prob:.2%}")
    else:
        st.success(f"Low Risk. Confidence: {1-prob:.2%}")