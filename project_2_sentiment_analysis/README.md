# project_2_sentiment_analysis

## Overview

This module performs sentiment analysis using a Logistic Regression model. It preprocesses text data, vectorizes it using TF-IDF, trains a classification model, and then evaluates its performance using a classification report and a confusion matrix visualization. The script assumes the input data (`df`) is already loaded into a pandas DataFrame format.

## Installation

The script utilizes several standard Python libraries. You can install them using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

(Note: `df` is expected to be a pandas DataFrame, though `pandas` itself is not explicitly imported in the provided snippet, it's a core dependency for the data structure.)

## Usage

This script is designed to be run in an environment where a pandas DataFrame named `df` is already loaded and available, containing the necessary `'selected_text'` and `'sentiment'` columns. It does not define functions or classes, and executes sequentially.

To use this module, ensure you have your data loaded into a DataFrame `df` and then run the script:

```python
# Example of how 'df' might be loaded/created (this part is NOT in the provided code,
# but illustrates the expected input for the script to run)
import pandas as pd
# Assuming df is loaded from a CSV, a database, or created programmatically
data = {
    'selected_text': [
        "I love this product, it's amazing!",
        "This is terrible, I'm so disappointed.",
        "It's okay, not great but not bad either.",
        "Absolutely fantastic experience!",
        "Worst purchase ever, completely useless.",
        "Feeling neutral about this one."
    ],
    'sentiment': [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral"
    ]
}
df = pd.DataFrame(data)

# --- The actual script starts here ---

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Data preprocessing
df = df.dropna(subset=['selected_text', 'sentiment']).reset_index(drop=True)

# Define features (X) and target (y)
X = df['selected_text']
y = df['sentiment']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=le.classes_, yticklabels=le.classes_
)
plt.title("Confusion Matrix - Sentiment Analysis")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
```

When executed, the script will print a classification report to the console and display a confusion matrix plot.

## Inputs & Outputs

### Inputs

The script expects a pre-existing pandas DataFrame named `df` in the global scope with the following columns:

*   **`selected_text`** (type: string): The text content on which sentiment analysis is to be performed. This serves as the feature variable (X).
*   **`sentiment`** (type: string): The categorical sentiment label for each text (e.g., 'positive', 'negative', 'neutral'). This serves as the target variable (y).

Rows with missing values in either `'selected_text'` or `'sentiment'` will be dropped.

### Outputs

1.  **Standard Output (Console)**:
    *   **Classification Report**: A detailed text report from `sklearn.metrics.classification_report` showing precision, recall, f1-score, and support for each sentiment class, along with overall accuracy, macro average, and weighted average.

2.  **Graphical Output (Matplotlib Plot)**:
    *   **Confusion Matrix**: A heatmap visualization using `seaborn` and `matplotlib.pyplot` that displays the number of true positive, true negative, false positive, and false negative predictions for each sentiment class. The plot window will open and display this figure.

## Explanation

The script performs the following steps for sentiment analysis:

1.  **Data Preprocessing**:
    *   Removes rows where `selected_text` or `sentiment` are missing.
    *   Defines `selected_text` as features (`X`) and `sentiment` as the target (`y`).
    *   Encodes the categorical `sentiment` labels into numerical format using `LabelEncoder`.

2.  **Data Splitting**:
    *   Splits the dataset into training (70%) and testing (30%) sets using `train_test_split`, ensuring an even distribution of sentiment classes across splits with `stratify=y_encoded` and `random_state=42` for reproducibility.

3.  **Text Vectorization**:
    *   Converts the text data (`selected_text`) into numerical TF-IDF (Term Frequency-Inverse Document Frequency) vectors using `TfidfVectorizer`.
    *   It removes English stop words and considers only the top 1000 most important features (words/n-grams).
    *   The vectorizer is `fit` on the training data and then used to `transform` both training and testing data.

4.  **Model Training**:
    *   Initializes a `LogisticRegression` model with `max_iter=1000` to ensure convergence.
    *   Trains the model using the TF-IDF vectorized training data (`X_train_vec`) and the encoded training labels (`y_train`).

5.  **Prediction**:
    *   Uses the trained model to make predictions (`y_pred`) on the TF-IDF vectorized test data (`X_test_vec`).

6.  **Model Evaluation**:
    *   Generates and prints a `classification_report` comparing the actual test labels (`y_test`) with the predicted labels (`y_pred`), providing key classification metrics.
    *   Computes a `confusion_matrix` from the actual and predicted labels.
    *   Visualizes the confusion matrix as a heatmap using `seaborn.heatmap`, making it easy to interpret the model's performance across different sentiment classes. The plot is displayed using `plt.show()`.

## License

[Add your chosen license here, e.g., MIT, Apache 2.0, etc.]