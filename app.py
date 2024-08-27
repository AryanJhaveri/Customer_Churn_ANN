import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_enocder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    label_encoder_geo = pickle.load(f)

# Streamlit app
st.set_page_config(page_title='Customer Churn Prediction', layout='wide', initial_sidebar_state='expanded')

# Custom CSS for better styling
st.markdown("""
    <style>
    .reportview-container {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #2E2E2E;
    }
    .css-1v0mbdj {
        background-color: #1E1E1E;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #6200EA;
        border-radius: 10px;
        font-size: 18px;
    }
    .stNumberInput input {
        border-radius: 5px;
        border: 1px solid #FFFFFF;
        padding: 5px;
        background-color: #333333;
        color: #FFFFFF;
    }
    .stSlider>div>div {
        background-color: #333333;
    }
    .stSelectbox div {
        background-color: #333333;
        color: #FFFFFF;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #FFFFFF;
    }
    h1 {
        color: #BB86FC;
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        text-align: center;
    }
    .prediction-result {
        font-size: 24px;
        font-weight: 600;
        color: #BB86FC;
    }
    .probability-result {
        font-size: 22px;
        color: #03DAC5;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Customer Churn Prediction')

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.write("Use the fields on the right to enter customer details and predict churn probability.")

# Columns for better layout
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)

with col2:
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    estimated_salary = st.number_input('Estimated Salary')
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.markdown(f'<div class="probability-result">Churn Probability: {prediction_proba:.2f}</div>', unsafe_allow_html=True)

if prediction_proba > 0.5:
    st.markdown('<div class="prediction-result">The customer is likely to churn.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="prediction-result">The customer is not likely to churn.</div>', unsafe_allow_html=True)
