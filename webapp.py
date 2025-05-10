import streamlit as st
import numpy as np
import pickle
import os

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and encoders from the same directory
with open(os.path.join(BASE_DIR, "fertilizer_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "soil_encoder.pkl"), "rb") as f:
    soil_encoder = pickle.load(f)
with open(os.path.join(BASE_DIR, "crop_encoder.pkl"), "rb") as f:
    crop_encoder = pickle.load(f)
with open(os.path.join(BASE_DIR, "fertilizer_encoder.pkl"), "rb") as f:
    fertilizer_encoder = pickle.load(f)

# Web app title
st.markdown("<h1 style='text-align: center;'>Fertilizer Recommendation System</h1>", unsafe_allow_html=True)
st.sidebar.header("Enter Environmental and Soil Details")

# Input fields
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, step=0.1)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
moisture = st.sidebar.number_input("Moisture (%)", min_value=0.0, max_value=100.0, step=0.1)

soil_options = soil_encoder.classes_.tolist()
soil_type = st.sidebar.selectbox("Soil Type", soil_options)

crop_options = crop_encoder.classes_.tolist()
crop_type = st.sidebar.selectbox("Crop Type", crop_options)

nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, step=1.0)
phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, step=1.0)
potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=200.0, step=1.0)

# Predict button
if st.sidebar.button("Predict Fertilizer"):
    try:
        # Encode inputs
        soil_encoded = soil_encoder.transform([soil_type])[0]
        crop_encoded = crop_encoder.transform([crop_type])[0]

        input_data = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded,
                                nitrogen, phosphorus, potassium]])

        # Make prediction
        prediction = model.predict(input_data)
        fertilizer_name = fertilizer_encoder.inverse_transform(prediction)[0]

        st.success(f"Recommended Fertilizer: *{fertilizer_name}*")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
