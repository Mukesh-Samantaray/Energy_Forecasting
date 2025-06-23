import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load and preprocess
df = pd.read_csv("PJMW_hourly.csv")
df['Datetime'] = pd.to_datetime(df['Datetime'])
daily_df = df.resample('D', on='Datetime').mean().reset_index()

# Add time-based features
daily_df['dayofweek'] = daily_df['Datetime'].dt.dayofweek
daily_df['month'] = daily_df['Datetime'].dt.month

# Add lag features
df_lag = daily_df.copy()
df_lag['lag1'] = df_lag['PJMW_MW'].shift(1)
df_lag['lag2'] = df_lag['PJMW_MW'].shift(2)
df_lag['lag7'] = df_lag['PJMW_MW'].shift(7)
df_lag.dropna(inplace=True)

# Train-test split
train_size = int(len(df_lag) * 0.8)
feature_cols = ['lag1', 'lag2', 'lag7', 'dayofweek', 'month']
X = df_lag[feature_cols]
y = df_lag['PJMW_MW']
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Streamlit UI
st.title("⚡ PJMW Multivariate Forecasting App")

# Forecast slider
forecast_days = st.slider("Select number of future days to forecast", min_value=1, max_value=30, value=7)

# Forecast future
last_known = df_lag.copy()
forecast_dates = []
forecast_values = []

for i in range(forecast_days):
    last_row = last_known.iloc[-1]
    lag1 = last_row['PJMW_MW']
    lag2 = last_known.iloc[-2]['PJMW_MW'] if len(last_known) > 1 else lag1
    lag7 = last_known.iloc[-7]['PJMW_MW'] if len(last_known) >= 7 else lag1

    next_date = last_known['Datetime'].iloc[-1] + pd.Timedelta(days=1)
    dayofweek = next_date.dayofweek
    month = next_date.month

    future_input = pd.DataFrame({
        'lag1': [lag1],
        'lag2': [lag2],
        'lag7': [lag7],
        'dayofweek': [dayofweek],
        'month': [month]
    })

    pred = rf.predict(future_input)[0]

    forecast_dates.append(next_date)
    forecast_values.append(pred)

    new_row = pd.DataFrame({
        'Datetime': [next_date],
        'PJMW_MW': [pred],
        'dayofweek': [dayofweek],
        'month': [month]
    })
    last_known = pd.concat([last_known, new_row], ignore_index=True)

# Forecast Table
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted PJMW_MW': forecast_values})
st.subheader("📅 Forecast Table")
st.dataframe(forecast_df)

# Download CSV
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast as CSV", csv, "forecast.csv", "text/csv")

# Actual vs Predicted Chart
st.subheader("📈 Actual vs Predicted (Test Set)")
chart_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_rf
}, index=daily_df['Datetime'].iloc[-len(y_test):])

fig, ax = plt.subplots()
chart_df.plot(ax=ax)
ax.set_ylabel("PJMW_MW")
ax.set_title("Actual vs Predicted Energy")
st.pyplot(fig)

# Observations
st.subheader("📊 Observations and Significance")

if forecast_values[0] < forecast_values[-1]:
    trend = "increasing"
elif forecast_values[0] > forecast_values[-1]:
    trend = "decreasing"
else:
    trend = "stable"

avg_forecast = np.mean(forecast_values)
max_forecast = np.max(forecast_values)
min_forecast = np.min(forecast_values)

st.markdown(f"""
- The forecast shows a **{trend} trend** over the next **{forecast_days} day(s)**.
- The **average forecasted load** is approximately **{avg_forecast:.2f} MW**.
- The **maximum** forecasted demand is **{max_forecast:.2f} MW**, and the **minimum** is **{min_forecast:.2f} MW**.
- Time-based features like **month** and **day of week** improve accuracy by capturing seasonal and weekly patterns.
""")
