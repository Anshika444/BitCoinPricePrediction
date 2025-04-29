
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")
st.title("📈 Bitcoin Price Prediction using Machine Learning")

# ----------------------
# Upload Files
# ----------------------
train_file = st.file_uploader("Upload Training CSV", type="csv")
test_file = st.file_uploader("Upload Test CSV", type="csv")

def preprocess(df):
    df = df.copy()
    df.replace("-", np.nan, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_index()
    df = df.fillna(method='ffill')
    return df

def add_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['return_1h'] = df['Close'].pct_change(1)
    df['volatility_2h'] = df['Close'].rolling(window=2).std()
    df['sma_2h'] = df['Close'].rolling(window=2).mean()
    df = df.fillna(method='bfill')
    return df

if train_file is not None and test_file is not None:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    features = ['Open', 'High', 'Low', 'Volume', 'Market Cap',
                'hour', 'dayofweek', 'return_1h', 'volatility_2h', 'sma_2h']
    target = 'Close'

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("🔍 Model Evaluation Metrics")
    st.write(f"**Mean Absolute Error:** {mae:.2f}")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    st.subheader("📊 Prediction vs Actual Plot")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index, y_test, label='Actual')
    ax.plot(y_test.index, y_pred, label='Predicted')
    ax.set_title("Bitcoin Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
else:
    st.info("Please upload both training and test CSV files.")
