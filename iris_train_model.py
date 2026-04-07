import streamlit as st
import pandas as pd
import joblib
import gdown
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset
file_id = "1FBCKZI4KKtlvZaY8XgvAcvLHo5We0ORF"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
output = "Iris.csv"

@st.cache_data
def load_data():
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

df = load_data()

# Prepare data of the Iris dataset
# features
X = df.drop(columns=["Id","Species"])
# target
y = df["Species"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)


# Save the model to a file
# This creates 'iris_model.joblib' in your current folder
joblib.dump(model, 'iris_model.joblib')

print("Model saved successfully!")
