#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
import joblib


# In[8]:


model=pickle.load(open('log7.pkl','rb'))


# In[9]:


scale= pickle.load(open('scale.pkl','rb'))


# In[21]:


df = pd.read_csv('diabetes.csv')  #Assign(8)


# In[23]:


import joblib

import joblib
from sklearn.preprocessing import StandardScaler
# Assuming previous code ran: df, model, scaler exist
joblib.dump(model, 'diabetes_model.joblib')
joblib.dump(scale, 'scaler.joblib')
joblib.dump(df.columns.tolist(), 'feature_names.joblib')  # For input order
print("Model and scaler saved!")

# Load model, scaler, features
def load_model():
    model = joblib.load('diabetes_model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()
    
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

# User inputs matching features
col1, col2 = st.columns(2)
with col1:
    Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    Glucose = st.number_input("Glucose", 0.0, 200.0, 120.0)
    BloodPressure = st.number_input("Blood Pressure", 0.0, 120.0, 70.0)
    SkinThickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
with col2:
    Insulin = st.number_input("Insulin", 0.0, 300.0, 80.0)
    BMI = st.number_input("BMI", 0.0, 60.0, 30.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    Age = st.number_input("Age", 20, 80, 30)

input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, 
                       DiabetesPedigreeFunction, Age]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

if st.button("Predict"):
    st.success(f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    st.info(f"Diabetes Probability: {probability:.2%}")


# In[ ]:





# In[ ]:





# In[ ]:




