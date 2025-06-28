# import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from src.data_loader import load_gold_data
from src.data_preprocessing import add_variables, prepare_data_for_training
from src.model_construction import load_latest_model, multi_step_predict, train_model, save_model
from src.model_evaluation import evaluate_model
import datetime

st.set_page_config(page_title="Gold Price Predictor", layout="wide")
st.title("📈 Gold Price Forecasting Dashboard")
st.subheader("Predict future gold prices using machine learning models")
st.markdown("Gold has become one of asset that is quite talked about recently since trade war from US and some countries, mainly China. And gold, as it called 'safe heaven', started to become the asset that is bought many people, especially Indonesia that made the price rose to All Time High (ATH). Through this project, I predict the gold price using XGBoost model ")

# --- Load and preprocess full data for visualization ---
df_visualization = load_gold_data(start_date="2015-01-01", end_date="2024-12-31", file_name='data_visualization', interval='1d')
df_training = load_gold_data(start_date="2022-01-01", end_date="2024-12-31", file_name='data_training', interval='1d')
df_training = add_variables(df_training)

min_date = pd.to_datetime("2015-01-01").date()
max_date = pd.to_datetime("2024-12-31").date()

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Options")

# Visualization date range
st.sidebar.markdown("Select the date range for visualization :")
date_start = st.sidebar.date_input("Select start date", value=min_date, min_value=min_date, max_value=max_date)
date_end = st.sidebar.date_input("Select end date", value=max_date, min_value=min_date, max_value=max_date)

if date_start > date_end:
    st.sidebar.error("❌ Start date must be before end date.")
    st.stop()

# Forecast horizon
forecast_days = st.sidebar.slider("📅 Days to Forecast", min_value=1, max_value=60, value=5)

# --- Filter and visualize ---
st.subheader("📊 Gold Price Movement")
st.markdown('''See the movement price of gold over the time (the available data is only the last ten years)''')
df_visualization = df_visualization[(df_visualization.index.date >= date_start) & (df_visualization.index.date <= date_end)]
st.line_chart(df_visualization["Close"], use_container_width=True)
date_start_formatted = date_start.strftime("%B %Y")
date_end_formatted = date_end.strftime("%B %Y")
st.markdown(f"From the chart above we can see that the price of gold is vary from time to time since {date_start_formatted} until {date_end_formatted}. But overall, the price is increasing.")

# --- Train/test split and model loading ---
X_train, y_train, X_test, y_test, X_forecast = prepare_data_for_training(df_training, k=3, test_size=0.2)
selected_columns = X_train.columns.tolist()
trained_model = train_model(X_train, y_train)

# --- Predict future prices using multi-step forecasting ---
forecast = multi_step_predict(
    model=trained_model,
    last_known_data=df_training.copy(),
    days=forecast_days,
    feature_creator=add_variables,
    selected_columns=selected_columns
)

# Save the model
save_model(trained_model, directory='models/')

st.subheader("🔮 Future Price Forecast")
st.markdown('''Forecasting gold price using data only from the year 2022 - 2024. You can choose how many days forward you want to forecast the price.''')

# create a DataFrame for the forecasted values
forecast_dates = pd.date_range(start=df_training.index.max() + pd.Timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({"Predicted Close": forecast.values}, index=forecast_dates)

# Now plot it
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_training.index[-30:], df_training["Close"].tail(30), label="Actual (Last 30 days)", marker="o")
ax.plot(forecast_df.index, forecast_df["Predicted Close"], label="Forecast", marker="x", color="orange")
ax.set_xlabel("Date")
ax.set_ylabel("Gold Price (USD)")
ax.set_title("Forecast vs Actual")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# --- Evaluation ---
# y_pred = model.predict(X_test)
eval_metrics = evaluate_model(y_test[:forecast_days], forecast.values)
# print(y_test)
st.subheader("📈 Evaluation Metrics")
st.markdown('''Evaluation metrics of the model using test data. The metrics used are Mean Absolute Error (MAE) and RMSE (Root Mean Squared Error).''')
st.write(eval_metrics)

st.success("✅ Forecast complete!")
