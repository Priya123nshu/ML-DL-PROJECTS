# project 9 disease diagnosis prediction

## Overview
This module implements a machine learning pipeline for predicting disease diagnoses based on medical record data. It loads a dataset, preprocesses both categorical features and target labels, then trains and evaluates two distinct classification models: Random Forest and Gradient Boosting. Finally, it provides performance metrics for both models and visualizes the most important features as identified by the Random Forest classifier.

## Installation
To run this script, you need to have Python installed along with the following libraries. You can install them using pip:

```bash
pip install pandas scikit-learn matplotlib
```

## Usage
This script is designed to be run directly. It expects a CSV file named `medical_records.csv` in the same directory as the script.

1.  **Prepare your data**: Ensure you have a CSV file named `medical_records.csv` with a column named "Diagnosis" and other columns representing medical features.
2.  **Run the script**: Execute the Python script from your terminal:

    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file).

The script will print accuracy scores and detailed classification reports for both models to the console. It will also display a bar chart showing the top 10 feature importances from the Random Forest model.

## Inputs & Outputs

### Inputs
*   **`medical_records.csv`**: A CSV file containing medical data.
    *   It **must** contain a column named `Diagnosis` which is used as the target variable.
    *   Other columns are treated as features. The script automatically handles categorical features by applying one-hot encoding (`pd.get_dummies`).

### Outputs
*   **Console Output**:
    *   Accuracy scores for Random Forest and Gradient Boosting models.
    *   Classification reports for both Random Forest and Gradient Boosting, including precision, recall, f1-score, and support for each diagnosis class.
*   **Graphical Output**:
    *   A Matplotlib plot titled "Top 10 Feature Importances (Random Forest)" displaying a bar chart of the ten most important features for predicting diagnosis, as determined by the Random Forest model. This plot will appear in a separate window.

## Explanation

The script follows a standard machine learning workflow:

1.  **Load Dataset**: Reads `medical_records.csv` into a pandas DataFrame.
2.  **Split Features and Target**: Separates the `Diagnosis` column as the target variable (`y`) and all other columns as features (`X`).
3.  **Encode Target Labels**: The `Diagnosis` column, which is likely categorical, is converted into numerical labels using `sklearn.preprocessing.LabelEncoder`. This is necessary for model training.
4.  **Encode Categorical Features**: Any categorical features within `X` are converted into a numerical format using one-hot encoding (`pd.get_dummies`). This creates new binary columns for each category, preventing the model from assuming ordinal relationships.
5.  **Train-Test Split**: The dataset is divided into training (80%) and testing (20%) sets. A `random_state` is set for reproducibility, and `stratify=y_encoded` ensures that the proportion of each diagnosis class is maintained in both training and test sets.
6.  **Random Forest Model**: An ensemble `RandomForestClassifier` is initialized with 200 estimators and trained on the training data. Predictions are then made on the test set.
7.  **Gradient Boosting Model**: A `GradientBoostingClassifier` is initialized with 200 estimators, a learning rate of 0.1, and a maximum depth of 3. It is trained and used to make predictions on the test set.
8.  **Evaluation**:
    *   The `accuracy_score` is calculated for both models.
    *   `classification_report` is generated for both models, providing a more detailed breakdown of performance metrics per class.
9.  **Feature Importance**: The feature importances from the trained Random Forest model are extracted. The top 10 most important features are identified, and a bar chart is generated using `matplotlib` to visualize their relative importance.

## License
This project is open-source and available under the [License Name] License. See the `LICENSE` file for more details.