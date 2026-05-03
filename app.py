import streamlit as st
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #667eea, #764ba2);
    color: white;
}
</style>
""", unsafe_allow_html=True)
import numpy as np
import pickle

# Title
st.title("Student Performance Prediction")

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Inputs
hours_studied = st.number_input("Hours Studied")
attendance = st.number_input("Attendance (%)")
sleep = st.number_input("Sleep Hours")

# Button
if st.button("Predict"):
    features = np.array([[hours_studied, attendance, sleep]])
    prediction = model.predict(features)
    
    st.success(f"Predicted Student Score: {round(prediction[0], 2)}")
