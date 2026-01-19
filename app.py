import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# Page Title
# -----------------------------
st.title("⚡ Power Consumption Prediction")
st.write("Linear Regression using real electricity dataset")

# -----------------------------
# Load Dataset
# -----------------------------
path = r"C:\Users\Aathira\Desktop\Electricity\powerconsumption.csv"
df = pd.read_csv(path)

# -----------------------------
# Show Dataset (optional)
# -----------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# -----------------------------
# Select Features & Target
# -----------------------------
X = df[['Temperature', 'Humidity', 'WindSpeed',
        'GeneralDiffuseFlows', 'DiffuseFlows']]
y = df['PowerConsumption_Zone1']

# -----------------------------
# Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Input Values")

temp = st.number_input("Temperature", value=25.0)
hum = st.number_input("Humidity", value=60.0)
wind = st.number_input("Wind Speed", value=4.0)
gdf = st.number_input("General Diffuse Flows", value=300.0)
dflo = st.number_input("Diffuse Flows", value=100.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Power Consumption"):
    user_data = np.array([[temp, hum, wind, gdf, dflo]])
    prediction = model.predict(user_data)

    st.success(f"🔮 Predicted Power Consumption (Zone 1): {prediction[0]:.2f}")
