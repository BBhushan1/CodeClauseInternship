import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


file_path = 'heart_disease_dataset.csv'
data = pd.read_csv(file_path)


data = data.dropna()


categorical_columns = ['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Chest Pain Type']
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.title('Heart Disease Risk Prediction')


age = st.number_input('Age', min_value=0, max_value=120, value=30)
gender = st.selectbox('Gender', options=['Male', 'Female'])
cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=300, value=120)
heart_rate = st.number_input('Heart Rate', min_value=0, max_value=200, value=70)
smoking = st.selectbox('Smoking', options=['Never', 'Former', 'Current'])
alcohol_intake = st.selectbox('Alcohol Intake', options=['None', 'Light', 'Moderate', 'Heavy'])
exercise_hours = st.number_input('Exercise Hours per Week', min_value=0, max_value=168, value=5)
family_history = st.selectbox('Family History', options=['No', 'Yes'])
diabetes = st.selectbox('Diabetes', options=['No', 'Yes'])
obesity = st.selectbox('Obesity', options=['No', 'Yes'])
stress_level = st.number_input('Stress Level', min_value=0, max_value=10, value=5)
blood_sugar = st.number_input('Blood Sugar', min_value=0, max_value=400, value=100)
exercise_induced_angina = st.selectbox('Exercise Induced Angina', options=['No', 'Yes'])
chest_pain_type = st.selectbox('Chest Pain Type', options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])


if st.button('Predict'):
    
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure],
        'Heart Rate': [heart_rate],
        'Smoking': [smoking],
        'Alcohol Intake': [alcohol_intake],
        'Exercise Hours': [exercise_hours],
        'Family History': [family_history],
        'Diabetes': [diabetes],
        'Obesity': [obesity],
        'Stress Level': [stress_level],
        'Blood Sugar': [blood_sugar],
        'Exercise Induced Angina': [exercise_induced_angina],
        'Chest Pain Type': [chest_pain_type]
    })

    for column in categorical_columns:
        input_data[column] = label_encoders[column].transform(input_data[column])

   
    input_data_scaled = scaler.transform(input_data)

  
    prediction = model.predict(input_data_scaled)
    risk = 'High' if prediction[0] == 1 else 'Low'
    
    st.write(f'Your risk of heart disease is: {risk}')
