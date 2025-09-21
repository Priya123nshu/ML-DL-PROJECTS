# project_7_customer_lifetime_valure_prediction

## Overview

This module implements a basic machine learning workflow to predict Customer Lifetime Value (CLV) using a Linear Regression model. It reads a dataset from a CSV file, preprocesses the data by dropping irrelevant or leakage columns, scales the features, splits the data into training and testing sets, trains a linear regression model, makes predictions, and evaluates the model's performance using standard regression metrics (MAE, RMSE, R²). Finally, it displays the coefficients of the trained model to infer feature importance.

## Installation

This project requires Python and the following libraries. You can install them using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

This module is designed to be run as a standalone Python script.
It expects a dataset named `customer_lifetime_value_prediction.csv` to be present in the same directory as the script.

To run the script:

```bash
python your_script_name.py
```

Replace `your_script_name.py` with the actual name of your Python file.

Upon execution, the script will print the evaluation metrics (MAE, RMSE, R²) and a table of feature coefficients.

## Inputs & Outputs

### Input

*   **`customer_lifetime_value_prediction.csv`**: A CSV file containing customer data, including features for prediction and a `predicted_clv` column as the target variable.
    *   **Required columns**:
        *   `predicted_clv`: The target variable to be predicted.
        *   Other numerical columns that serve as features after specified drops.

### Output

The script prints the following to the console:

*   **Evaluation Metrics**:
    *   Mean Absolute Error (MAE)
    *   Root Mean Squared Error (RMSE)
    *   R-squared (R²)
*   **Feature Importance**: A Pandas DataFrame showing the features and their corresponding Linear Regression coefficients, sorted in descending order of coefficient value.

Example output:

```
MAE: [calculated_mae_value]
RMSE: [calculated_rmse_value]
R²: [calculated_r2_value]

Feature Importance (Linear Regression Coefficients):
                Feature  Coefficient
...
[feature_name_1]    [coefficient_1]
[feature_name_2]    [coefficient_2]
...
```

## Explanation

The code performs the following steps:

1.  **Load Dataset**: Reads `customer_lifetime_value_prediction.csv` into a Pandas DataFrame.
2.  **Data Preprocessing**:
    *   Drops several columns identified as useless or potential data leakage: `"Unnamed: 0"`, `"Customer ID"`, `"p_not_alive"`, `"p_alive"`, `"error"`, `"actual30"`, `"predicted_purchases"`, `"expected_avg_sales_"`.
3.  **Features and Target Definition**:
    *   The `predicted_clv` column is designated as the target variable (`y`).
    *   All remaining columns form the feature set (`X`).
4.  **Feature Scaling**:
    *   `StandardScaler` is used to standardize the feature columns (`X_scaled`), ensuring that they have a mean of 0 and a standard deviation of 1. This is a common practice for linear models to prevent features with larger scales from dominating the learning process.
5.  **Train-Test Split**:
    *   The scaled features (`X_scaled`) and target (`y`) are split into training and testing sets.
    *   `30%` of the data is allocated for testing (`test_size=0.3`).
    *   `random_state=42` ensures reproducibility of the split.
6.  **Model Training**:
    *   A `LinearRegression` model is initialized and trained (`fit`) using the training data (`X_train`, `y_train`).
7.  **Predictions**:
    *   The trained model makes predictions (`predict`) on the unseen test feature set (`X_test`).
8.  **Model Evaluation**:
    *   The predictions (`y_pred`) are compared against the actual test target values (`y_test`) using three common regression metrics:
        *   **Mean Absolute Error (MAE)**: The average of the absolute differences between predictions and actual values.
        *   **Root Mean Squared Error (RMSE)**: The square root of the average of the squared differences between predictions and actual values.
        *   **R-squared (R²)**: The proportion of the variance in the dependent variable that is predictable from the independent variables.
9.  **Feature Importance**:
    *   The coefficients from the trained `LinearRegression` model are extracted.
    *   These coefficients are then paired with their corresponding feature names and sorted to show which features have the strongest positive or negative linear relationship with the target variable, according to the model.

## License

This project is open-sourced under the [LICENSE NAME] license. (e.g., MIT, Apache 2.0)