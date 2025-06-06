import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# File path
DATA_PATH = "IMDB-Movie-Dataset(2023-1951).csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Clean and convert 'year' column
    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    # Convert Rating to numeric
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna(subset=['Rating'])

    # Keep only needed columns (remove cast)
    df = df[['genre', 'year', 'director', 'Rating']].dropna()
    return df

@st.cache_data
def train_model(df):
    X = df.drop('Rating', axis=1)
    y = df['Rating']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Categorical columns for one-hot encoding
    categorical_cols = ['genre', 'director']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')  # 'year' will be passed as-is (numeric)

    # Model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, rmse

# Streamlit UI
st.title("🎬 Movie Rating Predictor")

try:
    df = load_data(DATA_PATH)
    model, rmse = train_model(df)

    st.markdown(f"**RMSE:** {rmse:.2f}")

    st.header("📽️ Predict a Movie Rating")
    genre = st.selectbox("Genre", df['genre'].unique())
    director = st.selectbox("Director", df['director'].unique())
    year = st.number_input("Year", min_value=1900, max_value=2050, value=2024)

    if st.button("Predict Rating"):
        if year > 2024:
            st.error("❌ Prediction not allowed for years after 2024.")
        else:
            input_df = pd.DataFrame([[genre, director, year]], columns=['genre', 'director', 'year'])
            predicted_rating = model.predict(input_df)[0]
            st.success(f"✅ Predicted Rating for {year}: {predicted_rating:.2f}")

except FileNotFoundError:
    st.error(f"❌ Error: File '{DATA_PATH}' not found. Please make sure it's in the correct location.")
except Exception as e:
    st.error(f"❌ Error: {e}")
