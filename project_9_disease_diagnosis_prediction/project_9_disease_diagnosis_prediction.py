```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 1. Load dataset (update the file path if needed)
df = pd.read_csv("medical_records.csv")

# 2. Split into features and target
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# 3. Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Encode categorical features (gender, symptoms, etc.)
X = pd.get_dummies(X, drop_first=True)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```