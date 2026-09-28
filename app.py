import streamlit as st
import numpy as np
import joblib
import os

# Page Configuration with Custom Icon
st.set_page_config(
    page_title="StructPredict | AI Structural Analyzer",
    page_icon="🏗️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Professional Aesthetic Custom CSS (Dark Glassmorphism)
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        width: 100%;
        border: none;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
    }
    .creator-badge {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 14px;
        color: #60a5fa;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 20px;
    }
    .metric-container {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 14px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section with Author Credit
st.markdown("🏗️ <h1 style='display:inline-block; font-size: 36px; font-weight: 800; color: #ffffff;'> StructPredict</h1>", unsafe_allow_html=True)
st.markdown("<div class='creator-badge'>✨ Designed & Developed by <b>Tahmeed Raiyean</b></div>", unsafe_allow_html=True)
st.markdown("<p style='color: #9ca3af; font-size: 16px; margin-top: -10px;'>Instant machine learning surrogate modeling for reinforced concrete column capacity envelopes ($P_n$ & $M_n$).</p>", unsafe_allow_html=True)
st.markdown("---")

# Load model with error safety
@st.cache_resource
def load_model():
    model_path = "struct_predict_model.pkl"
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        return str(e)

model_result = load_model()

if model_result is None:
    st.error("⚠️ Model file (`struct_predict_model.pkl`) not found in the directory! Please run your training script first.")
elif isinstance(model_result, str):
    st.error(f"⚠️ Error loading model: {model_result}")
else:
    model = model_result
    
    st.subheader("🔧 Section Geometric & Material Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        width_b = st.slider("Section Width b (mm)", min_value=250.0, max_value=600.0, value=350.0, step=10.0)
        fc = st.selectbox("Concrete Strength $f'_c$ (MPa)", options=[20, 25, 30, 35, 40, 50], index=2)
        steel_ratio = st.slider("Gross Steel Ratio (ρ_g)", min_value=0.01, max_value=0.04, value=0.02, step=0.005)
    
    with col2:
        depth_h = st.slider("Section Depth h (mm)", min_value=300.0, max_value=800.0, value=500.0, step=10.0)
        fy = st.selectbox("Steel Yield Strength $f_y$ (MPa)", options=[400, 500], index=1)

    st.markdown("---")

    if st.button("🚀 Predict Column Capacities"):
        try:
            input_features = np.array([[width_b, depth_h, fc, fy, steel_ratio]])
            prediction = model.predict(input_features)
            pn_pred = prediction[0][0]
            mn_pred = prediction[0][1]
            
            st.success("✅ Surrogate Model Inference Computed Successfully!")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Nominal Axial Capacity (Pn)", value=f"{pn_pred:.2f} kN")
            with res_col2:
                st.metric(label="Nominal Flexural Capacity (Mn)", value=f"{mn_pred:.2f} kNm")
                
            st.info("💡 **Engineering Note:** Predictions are derived via a Random Forest ensemble model trained on high-fidelity structural mechanics calculations.")
        except Exception as e:
            st.error(f"Prediction error: {e}")