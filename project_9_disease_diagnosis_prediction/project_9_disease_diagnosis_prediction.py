```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset (update filename if needed)
df = pd.read_csv("medical_records.csv")

# 2. Split into features and target
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# 3. Encode target labels (Diagnosis)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Encode categorical features (gender, symptoms, etc.)
X = pd.get_dummies(X, drop_first=True)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 6. Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 7. Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

# 8. Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_preds))

print(
    "\nRandom Forest Report:\n",
    classification_report(y_test, rf_preds, target_names=label_encoder.classes_),
)
print(
    "\nGradient Boosting Report:\n",
    classification_report(y_test, gb_preds, target_names=label_encoder.classes_),
)

# 9. Feature importance from Random Forest
importances = rf_model.feature_importances_
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 5))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), X.columns[indices[:10]], rotation=45)
plt.show()
```