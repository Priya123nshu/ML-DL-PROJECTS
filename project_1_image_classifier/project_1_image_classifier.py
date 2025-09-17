```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def show_sample_images(images, labels, n=5):
    """Display n sample digit images with labels."""
    fig, axes = plt.subplots(1, n, figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")
    plt.suptitle("Sample Digits from Dataset")
    plt.show()


def plot_confusion_matrix(cm):
    """Visualize confusion matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def main():
    # Load dataset
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    # Show sample images
    show_sample_images(digits.images, y, n=5)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM model
    svm_clf = SVC(kernel="rbf", gamma=0.001, C=10)
    svm_clf.fit(X_train, y_train)

    # Predictions
    y_pred = svm_clf.predict(X_test)

    # Evaluation
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)


if __name__ == "__main__":
    main()

```