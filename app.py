import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .app-card {
        background: linear-gradient(135deg, #fff7fb, #fdf2f8);
        padding: 28px;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        border: 1px solid #f3d6e4;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #2d2d44;
        margin-bottom: 6px;
    }
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #5c5c72;
        margin-bottom: 28px;
    }
    .result-box {
        background: #ffffff;
        border: 1px solid #f0cddd;
        border-radius: 16px;
        padding: 18px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
    }
    .result-title {
        font-size: 18px;
        color: #7a3d5c;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .result-value {
        font-size: 28px;
        font-weight: 800;
        color: #d63384;
    }
    </style>
""", unsafe_allow_html=True)

try:
    model = joblib.load("iris_model.joblib")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.markdown('<div class="app-card">', unsafe_allow_html=True)

st.markdown('<div class="title">🌸 Iris Species Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter the flower measurements below to identify the iris species.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", value=5.10, step=0.1, format="%.2f")
    petal_length = st.number_input("Petal Length (cm)", value=1.40, step=0.1, format="%.2f")

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", value=3.50, step=0.1, format="%.2f")
    petal_width = st.number_input("Petal Width (cm)", value=0.20, step=0.1, format="%.2f")

predict = st.button("Identify Species", use_container_width=True)

if predict:
    try:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        species_map = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }

        species = species_map.get(prediction, str(prediction))

        st.markdown(f"""
            <div class="result-box">
                <div class="result-title">Predicted Species</div>
                <div class="result-value">{species}</div>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown('</div>', unsafe_allow_html=True)
