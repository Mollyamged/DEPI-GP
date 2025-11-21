import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load("heart_model.pkl")
    label_encoders = joblib.load("encoders.pkl")
    return model, label_encoders

model, label_encoders = load_model()



# Dark mode styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* ✅ Force Predict button text to stay white */
    .stButton>button {
        background-color: #FF4B4B !important;
        color: white !important;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #FF6B6B !important;
        color: white !important;
    }

    /* ✅ Make prediction text always white */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white !important;
    }

    .healthy {
        background-color: #1A472A;
        border-left: 5px solid #2E8B57;
    }

    .risk {
        background-color: #5D1F1A;
        border-left: 5px solid #DC143C;
    }

    .sidebar .sidebar-content {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("❤️ Heart Disease Prediction")
    st.markdown("Enter patient information to assess the risk of heart disease")

with col2:
    st.image("https://media.tenor.com/91scJf-xrKEAAAAj/emoji-coraz%C3%B3n-humano.gif", width=150)

# Input form
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", options=["M", "F"])
    chest_pain_type = st.selectbox("Chest Pain Type",
                                   options=["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)",
                           min_value=90, max_value=200, value=120)
    cholesterol = st.slider("Cholesterol (mg/dl)",
                            min_value=100, max_value=400, value=200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                              options=[0, 1])
    resting_ecg = st.selectbox("Resting ECG",
                               options=["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate",
                       min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina",
                                   options=["N", "Y"])
    oldpeak = st.slider("Oldpeak",
                        min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    st_slope = st.selectbox("ST Slope",
                            options=["Up", "Flat", "Down"])

# Prediction button
if st.button("Predict Heart Disease Risk"):
    input_data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }
    
    df_input = pd.DataFrame([input_data])
    
    # Encode categorical columns
    cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    for col in cat_cols:
        df_input[col] = label_encoders[col].transform(df_input[col])
    
    # Predict
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]
    
    st.header("Prediction Results")
    
    if prediction == 0:
        st.markdown(f"""
        <div class="prediction-box healthy">
            <h3>✅ Low Risk of Heart Disease</h3>
            <p>The model predicts a <b>{probability:.1%}</b> probability of heart disease.</p>
            <p>Based on the input, the patient appears to have a lower risk profile.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box risk">
            <h3>⚠️ High Risk of Heart Disease</h3>
            <p>The model predicts a <b>{probability:.1%}</b> probability of heart disease.</p>
            <p>A medical consultation is strongly recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability bar
    st.subheader("Risk Probability")
    st.progress(float(probability))
    st.caption(f"Probability: {probability:.1%}")

# Sidebar info
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This application uses a machine learning model to predict the likelihood of heart disease.

    **Input Features:**
    - Age  
    - Sex  
    - Chest Pain Type  
    - Resting Blood Pressure  
    - Cholesterol  
    - Fasting Blood Sugar  
    - Resting ECG  
    - Maximum Heart Rate  
    - Exercise Angina  
    - Oldpeak  
    - ST Slope  

    **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

# Input explanation
with st.expander("Understanding Input Parameters"):
    st.markdown("""
    **Chest Pain Types:**
    - **ATA**: Atypical Angina  
    - **NAP**: Non-Anginal Pain  
    - **ASY**: Asymptomatic  
    - **TA**: Typical Angina  
    
    **Resting ECG Results:**
    - **Normal**: Normal  
    - **ST**: ST-T wave abnormality  
    - **LVH**: Left ventricular hypertrophy  
    
    **ST Slope:**
    - **Up**: Upsloping  
    - **Flat**: Flat  
    - **Down**: Downsloping  
    """)
