# Project_1_Image_classifier

## Overview

This module implements a basic image classification pipeline using a Support Vector Machine (SVM) on the `digits` dataset provided by scikit-learn. The script demonstrates data loading, visualizing sample images, splitting data into training and testing sets, feature scaling, training an SVM classifier, making predictions, and evaluating the model's performance through a classification report and a confusion matrix plot.

## Installation

This project requires the following Python libraries. You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

To run the image classification pipeline, simply execute the Python script:

```bash
python your_module_name.py
```

Upon execution, the script will:
1.  Load the `digits` dataset.
2.  Display 5 sample digit images from the dataset.
3.  Train an SVM classifier.
4.  Print a detailed classification report to the console.
5.  Display a confusion matrix plot of the model's performance.

### Example of internal function usage (for reference, not directly callable by user)

While designed to run directly, the module contains helper functions:

**`show_sample_images(images, labels, n=5)`**
Displays `n` sample images with their corresponding labels.
```python
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
show_sample_images(digits.images, digits.target, n=3)
# This will open a matplotlib window showing 3 sample digits.
```

**`plot_confusion_matrix(cm)`**
Visualizes a given confusion matrix.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import datasets

# ... (rest of main() logic to get cm) ...
digits = datasets.load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_clf = SVC(kernel="rbf", gamma=0.001, C=10)
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm)
# This will open a matplotlib window showing the confusion matrix.
```

## Inputs & Outputs

### Inputs

*   **Dataset**: The script loads the `digits` dataset internally using `sklearn.datasets.load_digits()`. It does not take any external input files or user-provided data.

### Outputs

*   **Graphical Output 1**: A `matplotlib` figure displaying `n` sample digit images with their true labels.
*   **Standard Output**: A comprehensive classification report printed to the console, including precision, recall, f1-score, and support for each class.
*   **Graphical Output 2**: A `matplotlib` figure visualizing the confusion matrix generated from the model's predictions on the test set.

## Explanation

The `main` function orchestrates the entire workflow:

1.  **Load Dataset**: The `digits` dataset, consisting of 8x8 pixel grayscale images of handwritten digits (0-9), is loaded using `sklearn.datasets.load_digits()`.
2.  **Display Samples**: The `show_sample_images` function is called to visualize the first 5 digit images from the dataset along with their labels.
3.  **Data Splitting**: The dataset is divided into training and testing sets using `train_test_split`, with 70% of the data allocated for training and 30% for testing. A `random_state` is set for reproducibility.
4.  **Feature Scaling**: `StandardScaler` is used to standardize the features (pixel values) by removing the mean and scaling to unit variance. This step is crucial for many machine learning algorithms, including SVMs, to perform optimally.
5.  **Model Training**: A Support Vector Classifier (`SVC`) is initialized with a Radial Basis Function (RBF) kernel, `gamma=0.001`, and `C=10`. The model is then trained on the scaled training data (`X_train`, `y_train`).
6.  **Prediction**: The trained SVM model makes predictions (`y_pred`) on the scaled test data (`X_test`).
7.  **Evaluation**:
    *   A `classification_report` is generated and printed, providing a detailed summary of the model's performance for each digit class.
    *   A `confusion_matrix` is computed comparing the true labels (`y_test`) with the predicted labels (`y_pred`).
    *   The `plot_confusion_matrix` function is then called to visualize this confusion matrix as a heatmap.

## License

This project is licensed under the [LICENSE NAME] - see the LICENSE.md file for details (if applicable).
```