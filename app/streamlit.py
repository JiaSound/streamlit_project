import streamlit as st
import pickle
import pandas as pd
import os

# Load model with correct path
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../models/diabetes_rf_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

# App title
st.title('Diabetes Prediction App')

# Input form
with st.form("prediction_form"):
    st.header('Enter Patient Information')
    
    col1, col2 = st.columns(2)
    
    with col1:
        glucose = st.number_input('Glucose Level (mg/dL)', min_value=0.0, step=0.1, value=100.0)
        bloodpressure = st.number_input('Blood Pressure (mm Hg)', min_value=0.0, step=0.1, value=70.0)
        insulin = st.number_input('Insulin Level (μU/mL)', min_value=0.0, step=0.1, value=30.0)
    
    with col2:
        bmi = st.number_input('BMI (kg/m²)', min_value=0.0, step=0.1, value=25.0)
        age = st.number_input('Age (years)', min_value=0, step=1, value=30)
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.001, value=0.5)
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame([[glucose, bloodpressure, insulin, bmi, age, dpf]],
                                columns=['glucose', 'bloodpressure', 'insulin', 'bmi', 'age', 'diabetespedigreefunction'])
        
        # Make prediction
        pred = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0][1]  # Get probability
        
        # Show result
        st.subheader('Prediction Result')
        
        if pred == 1:
            st.error(f'High risk of Diabetes ({pred_proba*100:.1f}% probability) 🚨')
            st.warning('Please consult with a healthcare professional')
        else:
            st.success(f'Low risk of Diabetes ({pred_proba*100:.1f}% probability) ✅')
            st.info('Maintain healthy lifestyle habits')
            
