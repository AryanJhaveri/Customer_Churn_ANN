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
st.set_page_config(page_title='Customer Churn Prediction', layout='wide')

# Custom CSS for better styling
st.markdown("""
    <style>
    .reportview-container {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .css-1v0mbdj {
        background-color: #1E1E1E;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #007ACC;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
    }
    .stNumberInput input {
        border-radius: 5px;
        border: 1px solid #FFFFFF;
        padding: 10px;
        background-color: #2A2A2A;
        color: #FFFFFF;
        font-size: 18px;
        width: 100%;
    }
    .stSlider>div>div {
        background-color: #2A2A2A;
    }
    .stSelectbox div {
        background-color: #2A2A2A;
        color: #FFFFFF;
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #FFFFFF;
        font-size: 18px;
        width: 100%;
    }
    h1 {
        color: #FFFFFF;
        font-family: 'Arial', sans-serif;
        font-weight: 900;
        text-align: center;
        font-size: 42px;
        margin-bottom: 20px;
    }
    .prediction-result {
        font-size: 36px;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 20px;
        text-align: center;
    }
    .probability-result {
        font-size: 32px;
        color: #FFD700;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Customer Churn Prediction')

# Columns for better layout
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', label_encoder_geo.categories_[0], help="Select the customer's country of residence.", key="geography")
    gender = st.selectbox('Gender', label_encoder_gender.classes_, help="Select the customer's gender.", key="gender")
    age = st.slider('Age', 18, 92, help="Select the customer's age.", key="age")
    tenure = st.slider('Tenure', 0, 10, help="Number of years the customer has been with the bank.", key="tenure")
    num_of_products = st.slider('Number of Products', 1, 4, help="Number of products the customer is using.", key="num_of_products")

with col2:
    balance = st.number_input('Balance', help="Enter the customer's account balance.", format="%f", step=0.01, key="balance")
    credit_score = st.number_input('Credit Score', help="Enter the customer's credit score.", format="%f", step=0.01, key="credit_score")
    estimated_salary = st.number_input('Estimated Salary', help="Enter the customer's estimated salary.", format="%f", step=0.01, key="estimated_salary")
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], help="Does the customer have a credit card?", key="has_cr_card")
    is_active_member = st.selectbox('Is Active Member', [0, 1], help="Is the customer an active member?", key="is_active_member")

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