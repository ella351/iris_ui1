import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Species Predictor")
st.markdown(
    "Enter the flower measurements below to predict the iris species using a trained machine learning model."
)

st.info("Tip: You can adjust the values using the + and - buttons.")

try:
    model = joblib.load("iris_model.joblib")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.markdown("### Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        value=5.10,
        step=0.1,
        format="%.2f"
    )
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        value=1.40,
        step=0.1,
        format="%.2f"
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        value=3.50,
        step=0.1,
        format="%.2f"
    )
    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        value=0.20,
        step=0.1,
        format="%.2f"
    )

st.markdown("")

if st.button("🔍 Identify Species", use_container_width=True):
    try:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        species_map = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica"
        }

        species_name = species_map.get(prediction, str(prediction))

        st.success(f"Predicted Species: {species_name}")

        if species_name == "Setosa":
            st.markdown("### 🌱 Result: Setosa")
            st.write("This flower is predicted to belong to the **Setosa** species.")
        elif species_name == "Versicolor":
            st.markdown("### 🌿 Result: Versicolor")
            st.write("This flower is predicted to belong to the **Versicolor** species.")
        elif species_name == "Virginica":
            st.markdown("### 🌸 Result: Virginica")
            st.write("This flower is predicted to belong to the **Virginica** species.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
