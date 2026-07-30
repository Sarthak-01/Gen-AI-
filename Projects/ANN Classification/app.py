import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: linear-gradient(160deg, #0a0e1a 0%, #141b2d 40%, #1a1f3a 100%);
        }
        .stApp::before {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(ellipse at 20% 50%, rgba(72, 187, 120, 0.06) 0%, transparent 50%),
                        radial-gradient(ellipse at 80% 20%, rgba(56, 178, 172, 0.06) 0%, transparent 50%),
                        radial-gradient(ellipse at 50% 80%, rgba(99, 102, 241, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        .main-title {
            text-align: center;
            background: linear-gradient(135deg, #48bb78, #38b2ac, #667eea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            text-align: center;
            color: rgba(148, 163, 184, 0.8);
            font-size: 0.95rem;
            font-weight: 400;
            margin-bottom: 2rem;
            letter-spacing: 0.2px;
        }
        .section-title {
            background: linear-gradient(135deg, #48bb78, #38b2ac);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 1.1rem;
            font-weight: 700;
            margin: 0.8rem 0 0.6rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            letter-spacing: 0.3px;
        }
        .result-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 20px;
            padding: 2rem 2.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 8px 32px rgba(0,0,0,0.25);
            margin-top: 1.5rem;
            position: relative;
            z-index: 1;
        }
        .result-label {
            color: rgba(148, 163, 184, 0.7);
            font-size: 0.85rem;
            font-weight: 500;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }
        .result-value {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
        }
        .result-value.danger {
            background: linear-gradient(135deg, #fc8181, #f56565);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .result-value.safe {
            background: linear-gradient(135deg, #68d391, #48bb78);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #48bb78, #38b2ac, #667eea);
            background-size: 200% 200%;
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.85rem 1.5rem;
            font-size: 1.05rem;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.3px;
            transition: all 0.4s ease;
            box-shadow: 0 4px 20px rgba(72, 187, 120, 0.3);
            position: relative;
            z-index: 1;
        }
        .stButton > button:hover {
            background-position: right center;
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(72, 187, 120, 0.45);
        }
        .stButton > button:active {
            transform: translateY(0);
        }
        div[data-testid="stSlider"] > div {
            padding-top: 0.3rem;
        }
        div[data-testid="stSlider"] label {
            color: rgba(203, 213, 225, 0.9) !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }
        .stSelectbox label, .stNumberInput label {
            color: rgba(203, 213, 225, 0.9) !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }
        .stSelectbox > div {
            border-radius: 12px !important;
        }
        .stSelectbox > div:focus-within {
            border-color: rgba(72, 187, 120, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(72, 187, 120, 0.15) !important;
        }
        .stSelectbox svg {
            fill: #ffffff !important;
        }
        .stNumberInput > div > div > input {
            background: #1e2538 !important;
            border: 1px solid rgba(255,255,255,0.15) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            caret-color: #ffffff !important;
            transition: border-color 0.2s ease;
        }
        .stNumberInput > div > div > input:focus {
            border-color: rgba(72, 187, 120, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(72, 187, 120, 0.15) !important;
        }
        .stNumberInput > div > div > input + div {
            color: #ffffff !important;
        }
        .stNumberInput button {
            background: #1e2538 !important;
            border-color: rgba(255,255,255,0.15) !important;
            color: #ffffff !important;
        }
        .stNumberInput button svg {
            fill: #ffffff !important;
        }
        .stNumberInput button:hover {
            background: rgba(72, 187, 120, 0.2) !important;
        }
        .stSlider > div > div {
            color: #ffffff !important;
        }
        div[data-testid="stThumbValue"] {
            color: #ffffff !important;
            font-weight: 600 !important;
            background: rgba(72, 187, 120, 0.3) !important;
            padding: 2px 8px !important;
            border-radius: 6px !important;
        }
        div[data-testid="stTickBar"] p {
            color: rgba(255,255,255,0.7) !important;
        }
        div[role="slider"] {
            background: linear-gradient(135deg, #48bb78, #38b2ac) !important;
            box-shadow: 0 2px 8px rgba(72, 187, 120, 0.3) !important;
        }
        .stAlert {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 12px !important;
            color: #cbd5e1 !important;
        }
        .stAlert > div > div {
            color: #cbd5e1 !important;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        hr {
            border-color: rgba(255,255,255,0.05) !important;
        }
    </style>
""", unsafe_allow_html=True)

model = load_model('ann_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.markdown('<div class="main-title">🔮 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter customer details below to predict churn probability</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">📍 Demographics</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
with col2:
    gender = st.selectbox('Gender', label_encoder_gender.classes_)

st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider('Age', 18, 92, 35)
    tenure = st.slider('Tenure (years)', 0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
with col2:
    credit_score = st.number_input('Credit Score', 300, 900, 650)
    balance = st.number_input('Account Balance ($)', 0.0, 250000.0, 50000.0, step=1000.0)
    estimated_salary = st.number_input('Estimated Salary ($)', 0.0, 200000.0, 75000.0, step=1000.0)

st.markdown('<div class="section-title">💳 Account Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
with col2:
    is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])

input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
    'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
    'EstimatedSalary': estimated_salary,
}

geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([pd.DataFrame([input_data]), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

if st.button('🔍 Predict Churn'):
    prediction = model.predict(input_data_scaled)
    prediction_probability = prediction[0][0]

    st.markdown('<div class="result-label">Churn Probability</div>', unsafe_allow_html=True)

    pct = prediction_probability * 100
    bar_color = "#ff6b6b" if prediction_probability > 0.5 else "#51cf66"
    bar_html = f"""
    <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 20px; width: 100%; margin: 0.5rem 0; overflow: hidden;">
        <div style="width: {pct:.1f}%; background: {bar_color}; height: 100%; border-radius: 10px; transition: width 0.5s ease;"></div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

    if prediction_probability > 0.5:
        st.markdown(f'<div class="result-value danger">⚠️ {pct:.1f}% — Likely to Churn</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top: 1rem; padding: 0.75rem 1rem; background: rgba(252,129,129,0.08); border: 1px solid rgba(252,129,129,0.15); border-radius: 12px; color: rgba(252,129,129,0.9); font-size: 0.85rem; font-weight: 500;">💡 This customer shows strong churn indicators. Consider retention strategies.</div>', unsafe_allow_html=True)
    else:
        stay_pct = 100 - pct
        st.markdown(f'<div class="result-value safe">✅ {stay_pct:.1f}% — Likely to Stay</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top: 1rem; padding: 0.75rem 1rem; background: rgba(104,211,145,0.08); border: 1px solid rgba(104,211,145,0.15); border-radius: 12px; color: rgba(104,211,145,0.9); font-size: 0.85rem; font-weight: 500;">💡 This customer appears loyal and engaged.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
