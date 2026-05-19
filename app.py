import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration ---
MODEL_DIR = 'model'

# --- 1. Load Model and Preprocessing Objects ---
@st.cache_resource
def load_model_artifacts():
    try:
        with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'top_features.pkl'), 'rb') as f:
            top_features = pickle.load(f)
        return model, scaler, label_encoder, top_features
    except FileNotFoundError:
        st.error(f"Error: Model artifacts not found in '{MODEL_DIR}'. Please run `train_model.py` first.")
        st.stop()

model, scaler, label_encoder, top_features = load_model_artifacts()

# --- 2. Streamlit UI Setup ---
st.set_page_config(page_title="Personality Type Classifier", layout="wide")
st.title("🧠 Personality Type Classifier")
st.write("Adjust the sliders to predict a personality type (Extrovert, Introvert, or Ambivert).")
st.markdown("--- ")

# --- 3. User Input (Sidebar) ---
with st.sidebar:
    st.header("Input Personality Traits")
    st.write("Adjust the sliders for the selected 12 key features:")

    # Dictionary to hold user inputs
    user_inputs = {}

    # Dynamically create sliders for each top feature
    for feature in top_features:
        # Assuming features are typically in a 0-10 range based on dataset inspection
        min_val = 0.0
        max_val = 10.0
        default_val = 5.0 # A reasonable default for a mid-point
        
        # You might want to get actual min/max/mean from your training data for more accurate sliders
        # For this example, we'll use a generic 0-10 range.

        user_inputs[feature] = st.slider(
            f"{feature.replace('_', ' ').title()}",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.1
        )

st.markdown("--- ")

# --- 4. Make Prediction Button ---
if st.button("Predict Personality Type"):    
    # Convert user inputs to a DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Ensure the order of columns matches the training data
    input_df = input_df[top_features]

    # --- 5. Preprocess Input ---
    # Apply the same scaler used during training
    scaled_input = scaler.transform(input_df)

    # --- 6. Predict Personality Type ---
    prediction_encoded = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # Decode the prediction back to original labels
    predicted_personality_type = label_encoder.inverse_transform(prediction_encoded)[0]

    # --- 7. Display Prediction Nicely ---
    st.subheader("Prediction Result:")
    st.success(f"The predicted personality type is: **{predicted_personality_type}**")

    # --- 8. Display Probability/Confidence Score ---
    st.subheader("Confidence Scores:")
    proba_df = pd.DataFrame({
        'Personality Type': label_encoder.classes_,
        'Confidence': prediction_proba[0]
    }).sort_values(by='Confidence', ascending=False)

    st.dataframe(proba_df.style.format({'Confidence': "{:.2%}"}), use_container_width=True)

    # Optionally, visualize probabilities
    st.bar_chart(proba_df.set_index('Personality Type'))

    st.markdown("--- ")
    st.info("Note: The prediction is based on a Logistic Regression model trained on synthetic data.")
