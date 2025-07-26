import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

# ----------------------------
# 🎯 App Configuration
# ----------------------------
st.title("📈 Stock Price Prediction App")
st.subheader("Powered by LSTM Neural Network")

# ----------------------------
# 📥 User Input
# ----------------------------
ticker = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")

# ----------------------------
# 📦 Load Historical Stock Data
# ----------------------------
start_date = "2000-01-01"
end_date = "2025-12-31"
data = yf.download(ticker, start=start_date, end=end_date)

# ----------------------------
# 📊 Show Summary Statistics
# ----------------------------
st.write("### Historical Data", data.describe())

# ----------------------------
# 📉 Visualizations
# ----------------------------
st.subheader(f"{ticker} Closing Price vs Time")
fig = plt.figure(figsize=(12, 4))
plt.plot(data['Close'])
st.pyplot(fig)

st.subheader(f"{ticker} Closing Price vs Time with 100 days Moving Average")
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(data.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader(f"{ticker} Closing Price vs Time with 200 & 100 days Moving Average")
ma200 = data.Close.rolling(200).mean()
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(data.Close)
plt.plot(ma200)
plt.plot(ma100)
st.pyplot(fig)

# ----------------------------
# ✂️ Data Preprocessing
# ----------------------------
training_data = pd.DataFrame(data["Close"][0:int(len(data)*0.70)])
testing_data = pd.DataFrame(data["Close"][int(len(data)*0.70):int(len(data))])

scalar = MinMaxScaler(feature_range=(0,1))
training_data_array = scalar.fit_transform(training_data)

# ----------------------------
# 🧠 Load Pre-trained Model
# ----------------------------
model = load_model("keras_model.h5")

# ----------------------------
# 🧩 Prepare Input for Testing
# ----------------------------
past_100_days_data = training_data.tail(100)
final_df = pd.concat([past_100_days_data, testing_data], ignore_index=True)
input_data = scalar.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# ----------------------------
# 📈 Predict on Test Data
# ----------------------------
y_predicted = model.predict(x_test)
scale_factor = 1 / scalar.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader(f"{ticker} Stock Price Prediction vs Original")
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Price")
plt.plot(y_predicted, label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)

# ----------------------------
# 🔮 Predict Next Day Price
# ----------------------------
last_100_days = final_df[-100:].values
last_100_scaled = scalar.transform(last_100_days)

X_input = []
X_input.append(last_100_scaled)
X_input = np.array(X_input)
X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

predicted_price = model.predict(X_input)
predicted_price = scalar.inverse_transform(predicted_price)
st.subheader(f"📉 Predicted Price for Next Day: {predicted_price[0][0]:.2f}")

# ----------------------------
# 🔁 Predict Future n Days
# ----------------------------

# ⚙️ Helper Function: Predict next n days recursively
def predict_future(model, last_sequence, n_days, scaler):
    future_preds = []
    current_input = last_sequence.reshape(1, -1, 1)

    for _ in range(n_days):
        next_pred = model.predict(current_input)[0][0]
        future_preds.append(next_pred)

        # Update current_input with the latest prediction
        current_input = np.append(current_input[:, 1:, :], [[[next_pred]]], axis=1)

    # Inverse scale the prediction back to original price range
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    return future_preds

# 🕹️ User Input for future days
st.sidebar.subheader("📆 Predict Future")
n_days = st.sidebar.slider("Select number of days to predict:", 1, 60, 7)

# Predict
future_preds = predict_future(model, last_100_scaled, n_days, scalar)

# 📅 Generate future dates
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

# 📊 Plot future prediction
st.subheader(f"📆 Future Prediction for Next {n_days} Days")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_dates, future_preds, marker='o', label='Future Prediction')
ax.set_title(f"{ticker} Forecast for Next {n_days} Days")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Price")
ax.grid(True)
ax.legend()
st.pyplot(fig)
