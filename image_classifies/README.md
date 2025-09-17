# `image_classifies`

## Overview
This module provides a basic workflow for image classification using a Support Vector Machine (SVM) model. It utilizes the handwritten digits dataset from scikit-learn, demonstrating data loading, visualization of sample images, data splitting, feature scaling, model training, prediction, and evaluation through a classification report and confusion matrix.

## Installation
This module requires the following Python libraries:
*   `numpy`
*   `matplotlib`
*   `scikit-learn`

You can install them using pip:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage
The `image_classifies` module can be run directly as a script to execute the full classification workflow, or its individual functions can be imported and utilized.

### Running the Full Workflow
To run the entire image classification process, including data loading, model training, and evaluation, simply execute the script:
```bash
python image_classifies.py
```

This will display plots of sample images and the confusion matrix, and print a classification report to the console.

### Using Individual Functions
You can import and use the `show_sample_images` and `plot_confusion_matrix` functions independently:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from image_classifies import show_sample_images, plot_confusion_matrix, main

# --- Example 1: Run the main workflow ---
print("Running the main classification workflow:")
main()

# --- Example 2: Display sample images ---
# Load the digits dataset for demonstration
digits = datasets.load_digits()
images = digits.images
labels = digits.target

print("\nDisplaying 3 sample images:")
show_sample_images(images, labels, n=3)
# Note: plt.show() is called internally by show_sample_images.
# If you integrate it into a larger script, you might manage plt.show() differently.

# --- Example 3: Plot a custom confusion matrix ---
# Create a dummy confusion matrix for demonstration
# In a real scenario, this would come from `confusion_matrix(y_true, y_pred)`
print("\nPlotting a dummy confusion matrix:")
dummy_cm = np.array([
    [150, 5, 2],
    [10, 140, 0],
    [3, 0, 160]
])
plot_confusion_matrix(dummy_cm)
# Note: plt.show() is called internally by plot_confusion_matrix.
```

## Inputs & Outputs

### Functions

#### `show_sample_images(images, labels, n=5)`
*   **Inputs**:
    *   `images` (numpy.ndarray): A 3D array of image data, where the first dimension is the number of samples, and the subsequent dimensions represent the height and width of each image (e.g., `digits.images` has shape `(n_samples, 8, 8)`).
    *   `labels` (numpy.ndarray): A 1D array of integer labels corresponding to each image (e.g., `digits.target`).
    *   `n` (int, optional): The number of sample images to display. Defaults to 5.
*   **Output**: Displays a Matplotlib figure showing `n` sample images with their associated labels. The figure is shown using `plt.show()`.

#### `plot_confusion_matrix(cm)`
*   **Inputs**:
    *   `cm` (numpy.ndarray): A 2D square NumPy array representing the confusion matrix.
*   **Output**: Displays a Matplotlib figure visualizing the confusion matrix with a color bar. The figure is shown using `plt.show()`.

### Script Execution (`main()` function)
When the module is run as a script (calling `main()`):
*   **Input**: None (the `digits` dataset is loaded internally from `sklearn.datasets`).
*   **Outputs**:
    *   A Matplotlib plot showing 5 sample images from the digits dataset.
    *   A Matplotlib plot displaying the confusion matrix of the trained SVM model's performance on the test set.
    *   A printed `Classification Report` to standard output, detailing precision, recall, f1-score, and support for each class.

## Explanation
The `image_classifies` module implements a standard machine learning pipeline for classifying handwritten digits:

1.  **Data Loading**: The `digits` dataset, a collection of 8x8 pixel grayscale images of handwritten digits (0-9), is loaded from `sklearn.datasets`. The images are represented as 64-feature vectors (`X`) and their corresponding integer labels (`y`).

2.  **Sample Visualization**: The `show_sample_images` function is used to display a few example images directly from the loaded dataset's `digits.images` to provide a visual understanding of the data.

3.  **Data Splitting**: The dataset is divided into training (70%) and testing (30%) sets using `train_test_split`. This ensures that the model is evaluated on data it has not seen during training, providing an unbiased estimate of its performance.

4.  **Feature Scaling**: A `StandardScaler` is applied to the features (`X_train` and `X_test`). This process transforms the data such that it has a mean of 0 and a standard deviation of 1. Scaling is crucial for many machine learning algorithms, including SVMs, as it helps prevent features with larger numerical ranges from dominating the learning process.

5.  **Model Training**: A Support Vector Classifier (`SVC`) is initialized with a Radial Basis Function (RBF) kernel (`kernel="rbf"`), `gamma=0.001`, and `C=10`. The SVM model is then trained on the scaled training data (`X_train`, `y_train`).

6.  **Prediction**: After training, the model makes predictions (`y_pred`) on the scaled test dataset (`X_test`).

7.  **Evaluation**:
    *   **Classification Report**: The `classification_report` from `sklearn.metrics` is printed, providing a detailed breakdown of the model's performance per class, including precision, recall, and f1-score.
    *   **Confusion Matrix**: A `confusion_matrix` is computed comparing the true labels (`y_test`) against the predicted labels (`y_pred`). This matrix is then visualized using the `plot_confusion_matrix` function, offering a clear graphical representation of where the model is making correct and incorrect classifications.

## License
```
[License Name]
[Year] [Copyright Holder]

[Full text of the license, e.g., MIT, Apache 2.0, GPLv3]
```