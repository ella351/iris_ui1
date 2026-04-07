import streamlit as st
import joblib
import numpy as np

# Title
st.title("🌸 Iris Species Predictor")
st.write("Enter your measurements below to identify the iris species using a trained Machine Learning model.")

# Load model
try:
    model = joblib.load("iris_model.joblib")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", value=5.10, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", value=1.40, step=0.1)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", value=3.50, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", value=0.20, step=0.1)

# Prediction
if st.button("Identify Species ➜"):
    try:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        species_map = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }

        st.success(f"Predicted Species: {species_map.get(prediction, prediction)}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
