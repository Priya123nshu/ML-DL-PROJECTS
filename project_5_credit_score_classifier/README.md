# project_5_credit_score_classifier

## Overview
This module implements a basic credit scoring classifier using a Logistic Regression model. It generates a small, simulated dataset of credit applicant information, preprocesses the data by encoding categorical features and scaling numerical features, trains a logistic regression model to predict loan default, and evaluates the model's performance using a classification report and a confusion matrix heatmap.

## Installation
To run this script, you need to have the following Python libraries installed. You can install them using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Usage
This module is designed to be run as a standalone Python script.
1.  Save the provided code as a Python file (e.g., `credit_classifier.py`).
2.  Execute the script from your terminal:

```bash
python credit_classifier.py
```

Upon execution, the script will:
*   Process the simulated credit applicant data.
*   Train a Logistic Regression model.
*   Print a classification report to the console.
*   Display a confusion matrix as a heatmap.

## Inputs & Outputs

### Inputs
The module uses a hardcoded, simulated dataset of credit applicant information. This dataset is a Python dictionary named `data` that is converted into a pandas DataFrame. It contains 20 entries with the following features:
*   `Age`: Applicant's age (integer).
*   `Income`: Applicant's annual income (integer).
*   `LoanAmount`: The amount of loan applied for (integer).
*   `CreditHistory`: Applicant's credit history ('Good' or 'Bad').
*   `EmploymentStatus`: Applicant's employment status ('Employed', 'Self-employed', 'Unemployed', 'Retired').
*   `Default`: Whether the applicant defaulted on a loan ('No' or 'Yes'). This is the target variable.

### Outputs
The script produces two main outputs:

1.  **Classification Report (Console Output)**:
    A detailed text report showing precision, recall, f1-score, and support for each class (0: No Default, 1: Default), along with accuracy, macro avg, and weighted avg.

    ```
    Classification Report:

                  precision    recall  f1-score   support

               0       0.50      0.50      0.50         4
               1       0.33      0.33      0.33         3

        accuracy                           0.43         7
       macro avg       0.42      0.42      0.42         7
    weighted avg       0.43      0.43      0.43         7
    ```
    (Note: Actual values may vary slightly due to random split, but structure will be consistent.)

2.  **Confusion Matrix (Graphical Output)**:
    A seaborn heatmap visualizing the confusion matrix, displayed in a matplotlib window. This plot shows the counts of true positives, true negatives, false positives, and false negatives.

    *   **Title**: "Confusion Matrix - Credit Scoring Model"
    *   **X-axis**: "Predicted" (labels: 'No Default', 'Default')
    *   **Y-axis**: "Actual" (labels: 'No Default', 'Default')

    ![Confusion Matrix Example](confusion_matrix_example.png)
    (Note: The image above is a placeholder; the script will generate a similar-looking heatmap based on its execution.)

## Explanation

The script follows a standard machine learning workflow:

1.  **Data Generation**: A small, predefined dictionary is used to create a pandas DataFrame, simulating credit applicant data.

2.  **Categorical Feature Encoding**:
    *   `LabelEncoder` from `sklearn.preprocessing` is used to convert categorical string features (`CreditHistory`, `EmploymentStatus`) into numerical representations.
    *   The `Default` target variable is also label-encoded (`No` -> 0, `Yes` -> 1).

3.  **Feature and Target Separation**:
    *   Features (`X`) are created by dropping the 'Default' column from the DataFrame.
    *   The target variable (`y`) is set to the 'Default' column.

4.  **Feature Scaling**:
    *   `StandardScaler` from `sklearn.preprocessing` is applied to the features (`X`) to standardize them. This ensures that features with larger numerical ranges do not dominate the model training process.

5.  **Data Splitting**:
    *   The `train_test_split` function from `sklearn.model_selection` divides the scaled data into training (70%) and testing (30%) sets. A `random_state` is set for reproducibility.

6.  **Model Training**:
    *   A `LogisticRegression` model from `sklearn.linear_model` is initialized and trained using the `fit` method on the training data (`X_train`, `y_train`).

7.  **Prediction and Evaluation**:
    *   The trained model makes predictions on the test set (`X_test`).
    *   `classification_report` from `sklearn.metrics` generates a text summary of the model's performance on the test data.
    *   `confusion_matrix` from `sklearn.metrics` calculates the confusion matrix, which is then visualized as a heatmap using `seaborn` and `matplotlib.pyplot` to provide a clear graphical representation of the model's true positive, true negative, false positive, and false negative rates.

## License
This project is open-source and available under the [License Name] License. (e.g., MIT, Apache 2.0, etc.).