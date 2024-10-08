import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


#Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_enocder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    label_encoder_geo = pickle.load(f)
    
## streamlit app

#st.title('Customer Churn Prediction')
st.markdown('<h1 style="text-align: center;">Customer Churn Prediction</h1>', unsafe_allow_html=True)

# User input
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
    
    
with col2:
    estimated_salary = st.number_input('Estimated Salary')
    tenure = st.slider('Tenure', 0, 10)
    num_of_products = st.slider('Number of Products', 1, 4)
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

st.markdown(f'<h2 style="text-align: center;">Churn Probability: {prediction_proba:.2f}</h2>', unsafe_allow_html=True)

if prediction_proba > 0.5:
    st.markdown(f'<h2 style="text-align: center;">The customer is likely to churn.</h2>', unsafe_allow_html=True)
else:
    st.markdown(f'<h2 style="text-align: center;">The customer is not likely to churn.</h2>', unsafe_allow_html=True)
