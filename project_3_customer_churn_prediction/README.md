# project_3 customer_churn_prediction

## Overview

This module implements a machine learning pipeline for customer churn prediction using Logistic Regression. It handles data loading, preprocessing of categorical and numerical features, data splitting, model training, and evaluation. The output includes a detailed classification report and a visual confusion matrix to assess the model's performance.

## Installation

This project requires the following Python libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Usage

To run this customer churn prediction model, ensure you have the dataset named `customer_churn_dataset-testing-master.csv` in the same directory as the script. Then, execute the Python file directly:

```bash
python your_script_name.py
```

Upon execution, the script will:
1. Load and preprocess the data.
2. Train a Logistic Regression model.
3. Make predictions on a test set.
4. Print a classification report to the console.
5. Display a graphical confusion matrix plot.

## Inputs & Outputs

### Inputs

*   **`customer_churn_dataset-testing-master.csv`**: A CSV file containing customer data.
    *   It is expected to contain a 'Churn' column as the target variable (0 for no churn, 1 for churn).
    *   Categorical features like 'Gender', 'Subscription Type', and 'Contract Length' are expected for Label Encoding.
    *   Other numerical columns will be treated as features for scaling and model training.

### Outputs

*   **Standard Output (Console)**:
    *   A `Classification Report` detailing precision, recall, f1-score, and support for both churn and no-churn classes.
*   **Graphical Output (Matplotlib/Seaborn Plot)**:
    *   A `Confusion Matrix` visualization, displayed in a new window, showing the counts of True Positives, True Negatives, False Positives, and False Negatives.

## Explanation

The script follows a standard machine learning workflow for classification tasks:

1.  **Data Loading**: The `customer_churn_dataset-testing-master.csv` file is loaded into a pandas DataFrame.
2.  **Feature Encoding**: `LabelEncoder` is applied to the categorical columns ('Gender', 'Subscription Type', 'Contract Length') to convert their text values into numerical representations, which is necessary for machine learning models.
3.  **Feature and Target Separation**: The dataset is split into features (`X`), which include all columns except 'Churn', and the target variable (`y`), which is the 'Churn' column itself.
4.  **Feature Scaling**: `StandardScaler` is used to standardize the features (`X`). This transforms the data so that it has a mean of 0 and a standard deviation of 1, which helps various machine learning algorithms, including Logistic Regression, perform better and converge faster.
5.  **Data Splitting**: The scaled data (`X_scaled`) and target (`y`) are divided into training and testing sets using `train_test_split`. 70% of the data is used for training, and 30% is reserved for testing the model's performance on unseen data. A `random_state` is set for reproducibility.
6.  **Model Training**: A `LogisticRegression` model is initialized and trained on the training data (`X_train`, `y_train`). The `solver='liblinear'` is specified, and `random_state` is set for consistent results.
7.  **Prediction**: The trained model then makes predictions (`y_pred`) on the test set (`X_test`).
8.  **Model Evaluation**:
    *   **Classification Report**: The `classification_report` function provides a comprehensive summary of the model's performance, including precision, recall, F1-score, and support for each class (churn/no churn).
    *   **Confusion Matrix**: A `confusion_matrix` is computed to show the number of correct and incorrect predictions made by the classification model compared to the actual outcomes. This matrix is then visualized using `seaborn.heatmap` for easy interpretation.

## License

[Add your chosen license here, e.g., MIT, Apache 2.0, etc.]