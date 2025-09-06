import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="🧠 Disease Prediction AI", 
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# -------------------
# Load model and encoder
# -------------------
@st.cache_resource
def load_artifacts():
    model = load_model("disease_model.h5")
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_artifacts()

# -------------------
# Custom CSS for styling
# -------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1565c0;
        transform: scale(1.05);
    }
    .prediction-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .symptom-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.3rem 0.8rem;
        border-radius: 16px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .probability-bar {
        height: 10px;
        background-color: #e0e0e0;
        border-radius: 5px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .probability-fill {
        height: 100%;
        background-color: #42a5f5;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------
# Streamlit UI
# -------------------
st.markdown('<h1 class="main-header">🧠 AI Disease Prediction from Symptoms</h1>', unsafe_allow_html=True)

# Introduction
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    Welcome to our AI-powered disease prediction tool. 
    Simply select the symptoms you're experiencing from the list below, 
    and our advanced machine learning model will predict the most likely diseases.
    """)
    
    st.info("💡 **Tip**: Start typing in the symptoms selector to quickly find your symptoms.")

with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=150)

# Load symptoms list
with open("symptoms_list.txt", "r") as f:
    symptoms = [line.strip() for line in f.readlines()]

# Sidebar for additional info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This AI model has been trained on medical data to predict diseases based on symptoms.
    
    **Disclaimer**: This tool is for informational purposes only and is not a substitute for professional medical advice.
    Always consult a healthcare provider for proper diagnosis and treatment.
    """)
    
    st.header("📊 Statistics")
    st.metric("Number of Symptoms", len(symptoms))
    st.metric("Number of Diseases", len(le.classes_))

# Select symptoms
st.markdown('<div class="sub-header">Select Symptoms</div>', unsafe_allow_html=True)
selected_symptoms = st.multiselect(
    "Choose symptoms:",
    symptoms,
    placeholder="Start typing to search for symptoms...",
    label_visibility="collapsed"
)

# Display selected symptoms as tags
if selected_symptoms:
    st.markdown("**Selected Symptoms:**")
    tags_html = "".join([f'<span class="symptom-tag">{symptom}</span>' for symptom in selected_symptoms])
    st.markdown(tags_html, unsafe_allow_html=True)

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Predict Disease", use_container_width=True)

if predict_btn:
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Prepare input
        x_input = np.zeros(len(symptoms))
        for s in selected_symptoms:
            if s in symptoms:
                idx = symptoms.index(s)
                x_input[idx] = 1
        
        x_input = np.expand_dims(x_input, axis=0)

        # Prediction
        with st.spinner("Analyzing symptoms..."):
            preds = model.predict(x_input)
            top_indices = preds[0].argsort()[-3:][::-1]
            top_diseases = le.inverse_transform(top_indices)
            top_probs = preds[0][top_indices]

        # Display results
        st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
        
        for i, (disease, prob) in enumerate(zip(top_diseases, top_probs)):
            with st.container():
                st.markdown(f'<div class="prediction-card">', unsafe_allow_html=True)
                
                # Display rank with different emojis
                rank_emoji = ["🥇", "🥈", "🥉"][i]
                st.markdown(f"### {rank_emoji} {disease}")
                
                # Probability percentage
                st.markdown(f"**{prob*100:.2f}% probability**")
                
                # Visual probability bar
                st.markdown('<div class="probability-bar">', unsafe_allow_html=True)
                st.markdown(f'<div class="probability-fill" style="width: {prob*100}%"></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Additional information
        st.info("💡 The predictions are based on statistical patterns and should be verified by a medical professional.")