```python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. Fetch historical stock data
# -------------------------------
ticker = "AAPL"
start_date = "2022-01-01"
end_date = "2023-01-01"

data = yf.download(ticker, start=start_date, end=end_date)

# Keep only the closing prices
data = data[['Close']]

# -------------------------------
# 2. Create lag features
# -------------------------------
# Use the past N days to predict the next day's closing price
N = 5
for i in range(1, N + 1):
    data[f'Close_lag_{i}'] = data['Close'].shift(i)

# Remove rows with missing values
data.dropna(inplace=True)

# -------------------------------
# 3. Prepare features and target
# -------------------------------
feature_columns = [f'Close_lag_{i}' for i in range(1, N + 1)]
X = data[feature_columns]   # Features: past N days
y = data['Close']           # Target: today’s close

# -------------------------------
# 4. Split data into training and testing sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # Important for time series
)

# -------------------------------
# 5. Train the linear regression model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 6. Make predictions
# -------------------------------
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# -------------------------------
# 7. Visualize actual vs predicted prices
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label='Actual Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Price', color='orange', linestyle='--')
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Closing Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```