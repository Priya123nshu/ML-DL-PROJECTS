# project_4_stock_price_prediction

## Overview
This Python script demonstrates a basic approach to predicting stock prices using historical closing prices and a Linear Regression model. It fetches historical stock data for a specified ticker, creates lag features based on past closing prices, trains a linear regression model, makes predictions, and visualizes the actual vs. predicted prices.

## Installation
To run this script, you need to install the following Python libraries:

```bash
pip install yfinance pandas scikit-learn matplotlib
```

## Usage
This script is designed to be run directly. It fetches data, trains a model, makes predictions, and displays the results upon execution.

1.  **Save the code**: Save the provided Python code as `project_4_stock_price_prediction.py`.
2.  **Run the script**: Execute the script from your terminal.
    ```bash
    python project_4_stock_price_prediction.py
    ```

The script will print the Mean Squared Error of the model's predictions on the test set and display a plot showing the actual and predicted stock prices.

### Configuration
You can modify the following variables within the script to change the stock, date range, or the number of lag features:

*   `ticker`: The stock ticker symbol (e.g., `"AAPL"`, `"MSFT"`).
*   `start_date`: The start date for fetching historical data (e.g., `"2022-01-01"`).
*   `end_date`: The end date for fetching historical data (e.g., `"2023-01-01"`).
*   `N`: The number of past days' closing prices to use as features for prediction (e.g., `5`).

**Example of modifying parameters:**

```python
# ... (rest of the code)

# -------------------------------
# 1. Fetch historical stock data
# -------------------------------
ticker = "GOOGL"          # Change ticker to Google
start_date = "2023-01-01"
end_date = "2024-01-01"
# ... (rest of the code)

# Use the past 10 days to predict the next day's closing price
N = 10
# ... (rest of the code)
```

## Inputs & Outputs

### Inputs
The script takes the following parameters defined internally:
*   `ticker` (string): The stock symbol (e.g., "AAPL").
*   `start_date` (string): The start date for data fetching in "YYYY-MM-DD" format.
*   `end_date` (string): The end date for data fetching in "YYYY-MM-DD" format.
*   `N` (integer): The number of preceding days' closing prices to use as input features for prediction.

### Outputs
Upon execution, the script produces:
*   **Console Output**:
    *   `Mean Squared Error on Test Set: [value]` (float): The mean squared error between the actual and predicted closing prices on the test dataset.
*   **Graphical Output**:
    *   A `matplotlib` plot titled "[Ticker] Stock Price Prediction" displaying two lines:
        *   "Actual Price" (blue): The true closing prices from the test set.
        *   "Predicted Price" (orange, dashed): The prices predicted by the Linear Regression model for the test set.

## Explanation

The script follows a standard machine learning pipeline for time series prediction:

1.  **Fetch Historical Stock Data**: It uses the `yfinance` library to download historical stock data for the specified `ticker` and date range. Only the 'Close' price is retained for further processing.

2.  **Create Lag Features**: To enable time-series prediction, `N` lag features are created. For each day, `Close_lag_1` represents the closing price from the previous day, `Close_lag_2` from two days ago, and so on, up to `Close_lag_N`. Rows containing `NaN` values (due to shifting at the beginning of the dataset) are dropped.

3.  **Prepare Features and Target**:
    *   `X` (features) consists of the `N` lag columns (`Close_lag_1` to `Close_lag_N`).
    *   `y` (target) is the current day's `Close` price.

4.  **Split Data into Training and Testing Sets**: The data is split into training (80%) and testing (20%) sets using `train_test_split`. Crucially, `shuffle=False` is used to maintain the chronological order, which is essential for time series data.

5.  **Train the Linear Regression Model**: A `LinearRegression` model from `scikit-learn` is initialized and trained (`fit`) using the training features (`X_train`) and their corresponding target values (`y_train`).

6.  **Make Predictions**: The trained model then makes predictions (`predict`) on the unseen test features (`X_test`).

7.  **Evaluate the Model**: The performance of the model is evaluated using the Mean Squared Error (`mean_squared_error`) between the actual `y_test` values and the `y_pred` values. The MSE is printed to the console.

8.  **Visualize Actual vs. Predicted Prices**: `matplotlib` is used to generate a plot comparing the actual closing prices from the test set against the model's predicted closing prices over the same period. This visual representation helps in understanding the model's performance.

## License
[Add your chosen license here, e.g., MIT License, Apache 2.0, etc.]