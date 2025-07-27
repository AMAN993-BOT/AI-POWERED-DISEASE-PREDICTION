import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved model and encoders
model = joblib.load("naive_bayes_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")
feature_order = joblib.load("feature_order.pkl")

# Page configuration
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 AI-Powered Disease Predictor")
st.markdown("Check potential diseases based on symptoms using a Naive Bayes ML model.")

# Initialize session state
if "user_input" not in st.session_state:
    st.session_state.user_input = {feature: "No" for feature in feature_names}

# Form input for each symptom
st.subheader("Select Symptoms")
for feature in sorted(feature_names):
    st.session_state.user_input[feature] = st.selectbox(
        feature.replace("_", " ").capitalize(),
        ["No", "Yes"],
        index=0 if st.session_state.user_input[feature] == "No" else 1,
        key=feature
    )

# Convert user input to model format
input_data = {
    feature: 1 if val == "Yes" else 0
    for feature, val in st.session_state.user_input.items()
}
input_df = pd.DataFrame([input_data])[feature_order]

# Layout: Predict and Reset buttons
col1, col2 = st.columns(2)

# Predict Button
if col1.button("🔍 Predict Disease"):
    if any(val == "Yes" for val in st.session_state.user_input.values()):
        pred_encoded = model.predict(input_df)[0]
        disease = label_encoder.inverse_transform([pred_encoded])[0]

        st.success(f"✅ You might be suffering from: **{disease}**")

        # Show top 5 predictions
        st.subheader("Top 5 Possible Diseases")
        probs = model.predict_proba(input_df)[0]
        top_indices = np.argsort(probs)[::-1][:5]
        top_diseases = label_encoder.inverse_transform(top_indices)
        top_probs = probs[top_indices]

        for d, p in zip(top_diseases, top_probs):
            st.write(f"- {d}: **{p*100:.2f}%**")
    else:
        st.warning("⚠️ Please select at least one symptom.")

# Reset Button
if col2.button("🔁 Reset Symptoms"):
    for feature in feature_names:
        st.session_state.user_input[feature] = "No"
    st.success("🧹 All symptoms have been reset!")
    st.rerun()

# Disclaimer
st.markdown("---")
st.info("This tool is for educational purposes only. Please consult a medical professional for real diagnosis.")
