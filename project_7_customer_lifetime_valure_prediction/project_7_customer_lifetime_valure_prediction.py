```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("customer_lifetime_value_prediction.csv")

# Drop useless / leakage columns
df = df.drop(columns=[
    "Unnamed: 0",        # index
    "Customer ID",       # identifier
    "p_not_alive",       # derived probability
    "p_alive",           # derived probability
    "error",             # difference between prediction & actual
    "actual30",          # future outcome, not available at training
    "predicted_purchases",
    "expected_avg_sales_"
])

# -------------------------------
# 2. Features and Target
# -------------------------------
X = df.drop(columns=["predicted_clv"])
y = df["predicted_clv"]

# -------------------------------
# 3. Scale features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# -------------------------------
# 5. Fit Linear Regression
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 6. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 7. Evaluation
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# -------------------------------
# 8. Optional: Check feature importance (coefficients)
# -------------------------------
coefficients = pd.DataFrame({
    "Feature": df.drop(columns=["predicted_clv"]).columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance (Linear Regression Coefficients):")
print(coefficients)
```