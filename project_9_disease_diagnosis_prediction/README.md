```markdown
# project 9 disease diagnosis prediction

## Overview

This Python script is designed for the initial data preparation phase of a disease diagnosis prediction project. It loads medical record data from a CSV file, performs essential preprocessing steps such as encoding categorical features and the target variable, and then splits the processed data into training and testing sets. This prepares the data for subsequent machine learning model training.

## Installation

This script requires the following Python libraries. You can install them using pip:

```bash
pip install pandas scikit-learn
```

## Usage

To use this script, ensure you have a CSV file named `medical_records.csv` in the same directory as the script. The CSV file must contain a column named "Diagnosis", which will be used as the target variable.

Simply run the script:

```bash
python your_script_name.py
```

Replace `your_script_name.py` with the actual name you save the Python code as (e.g., `prepare_data.py`).

Upon execution, the script will print the shapes of the generated training and testing datasets for features and the target.

```
X_train shape: (rows, columns)
X_test shape: (rows, columns)
y_train shape: (rows,)
y_test shape: (rows,)
```

## Inputs & Outputs

### Inputs

*   **`medical_records.csv`**: A CSV file located in the same directory as the script.
    *   It must contain a column named `"Diagnosis"` which will serve as the target variable.
    *   Other columns will be treated as features. Categorical features (non-numeric) will be automatically one-hot encoded.

### Outputs

*   **Console Output**:
    *   The script prints the shapes of the resulting `X_train`, `X_test`, `y_train`, and `y_test` arrays to the console after the train-test split.
*   **In-memory Data**:
    *   After execution, the following preprocessed datasets are available in memory:
        *   `X_train`: Training features (Pandas DataFrame, one-hot encoded).
        *   `X_test`: Testing features (Pandas DataFrame, one-hot encoded).
        *   `y_train`: Encoded training target labels (NumPy array).
        *   `y_test`: Encoded testing target labels (NumPy array).
        *   `label_encoder`: The `LabelEncoder` instance used to transform the 'Diagnosis' column, which can be used to inverse transform predictions.

## Explanation

The script performs the following sequential steps:

1.  **Load Dataset**: Reads the `medical_records.csv` file into a Pandas DataFrame.
2.  **Split Features and Target**: Separates the DataFrame into features (`X`) by dropping the "Diagnosis" column, and the target variable (`y`) by selecting the "Diagnosis" column.
3.  **Encode Target Labels**: Applies `LabelEncoder` to the `y` (Diagnosis) column to convert categorical diagnosis names into numerical labels (e.g., 'Flu' -> 0, 'Cold' -> 1). This encoded target is stored in `y_encoded`.
4.  **Encode Categorical Features**: Utilizes `pd.get_dummies` to perform one-hot encoding on all categorical columns within the `X` (features) DataFrame. The `drop_first=True` argument is used to avoid multicollinearity.
5.  **Train-Test Split**: Divides the preprocessed features (`X`) and encoded target (`y_encoded`) into training and testing sets using `train_test_split`.
    *   `test_size=0.2`: 20% of the data is allocated for the test set.
    *   `random_state=42`: Ensures reproducibility of the split.
    *   `stratify=y_encoded`: Ensures that the proportion of each diagnosis category is the same in both the training and testing sets as it is in the original dataset.
6.  **Print Shapes**: Finally, it prints the dimensions (rows, columns) of the resulting training and testing feature sets (`X_train`, `X_test`) and target sets (`y_train`, `y_test`).

## License

[Add your chosen license here, e.g., MIT, Apache 2.0, etc.]
```