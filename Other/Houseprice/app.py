import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    # Replace with your dataset path (download from Kaggle)
    df = pd.read_csv("train.csv")  # Kaggle House Prices dataset
    return df

df = load_data()
st.title("🏡 House Price Prediction (ML + Streamlit)")

st.write("Dataset Preview:")
st.dataframe(df.head())

# ---------------------------
# Data Preprocessing
# ---------------------------
# Select a few important features for simplicity
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
target = "SalePrice"

data = df[features + [target]].dropna()

X = data[features]
y = data[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

st.sidebar.header("Enter House Details")
# ---------------------------
# User Inputs
# ---------------------------
def user_input_features():
    OverallQual = st.sidebar.slider('Overall Quality (1-10)', 1, 10, 5)
    GrLivArea = st.sidebar.slider('Above grade living area (sq ft)', 500, 4000, 1500)
    GarageCars = st.sidebar.slider('Garage capacity (cars)', 0, 4, 1)
    TotalBsmtSF = st.sidebar.slider('Total Basement area (sq ft)', 0, 3000, 500)
    FullBath = st.sidebar.slider('Full Bathrooms', 0, 4, 2)
    YearBuilt = st.sidebar.slider('Year Built', 1872, 2010, 1990)
    data = {
        'OverallQual': OverallQual,
        'GrLivArea': GrLivArea,
        'GarageCars': GarageCars,
        'TotalBsmtSF': TotalBsmtSF,
        'FullBath': FullBath,
        'YearBuilt': YearBuilt
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ---------------------------
# Prediction
# ---------------------------
prediction = model.predict(input_df)

st.subheader('Predicted House Price')
st.success(f"${prediction[0]:,.2f}")

# ---------------------------
# Model Performance
# ---------------------------
st.subheader("Model R² Score on Test Set")
st.write(f"R²: {model.score(X_test, y_test):.2f}")
