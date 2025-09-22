# project 8 document clustering system

## Overview

This Python script demonstrates a basic document clustering system using a combination of text vectorization, machine learning clustering, and dimensionality reduction for visualization. It takes a predefined set of sample documents, converts them into numerical feature vectors using TF-IDF, then groups them into clusters using the K-Means algorithm. Finally, it uses Principal Component Analysis (PCA) to reduce the data's dimensions for a 2D scatter plot visualization of the clusters and prints the cluster assignment for each document.

## Installation

This script requires the following Python libraries. You can install them using pip:

```bash
pip install scikit-learn matplotlib
```

## Usage

This module is designed to be run as a standalone script. It contains hardcoded sample documents and will immediately perform clustering and display the results upon execution.

1.  Save the code as a Python file (e.g., `document_clustering.py`).
2.  Run the script from your terminal:

    ```bash
    python document_clustering.py
    ```

Upon execution, a matplotlib window will pop up displaying the clustered documents, and the console will output the cluster assignment for each original document.

## Inputs & Outputs

### Inputs

The script's input is a hardcoded list of strings named `documents`:

```python
documents = [
    "Apple released a new iPhone today.",
    "The stock market saw major gains.",
    "Google announced a new Android update.",
    "Investors are optimistic about tech stocks.",
    "iPhones are selling fast this year.",
    "The economy is recovering from the pandemic.",
    "Samsung's Galaxy phones compete with iPhones.",
    "Financial experts predict more growth in the market.",
    "Android phones have improved camera features.",
    "Inflation rates remain a concern for investors."
]
```

### Outputs

1.  **Graphical Plot**: A matplotlib scatter plot window titled "Document Clustering with K-Means". This plot visualizes the documents in a 2-dimensional space (reduced by PCA), with each point colored according to its assigned cluster.

    *   X-axis: PCA Component 1
    *   Y-axis: PCA Component 2
    *   Legend: Shows the labels for each cluster.

2.  **Console Output**: Prints each original document along with the cluster it was assigned to.

    ```
    Cluster 1: Apple released a new iPhone today.
    Cluster 3: The stock market saw major gains.
    ...
    ```

## Explanation

The script follows a sequence of common steps in document clustering:

1.  **TF-IDF Vectorization**:
    *   `TfidfVectorizer` is used to convert the raw text `documents` into a matrix of TF-IDF features. This process considers the importance of a word in a document relative to the entire set of documents, while also removing common English stop words (e.g., "the", "is", "a") which usually don't carry much meaning for clustering.
    *   The output `X` is a sparse matrix representing the numerical features of each document.

2.  **K-Means Clustering**:
    *   The `KMeans` algorithm is initialized to find `n_clusters = 3` distinct groups within the vectorized documents. `random_state=42` ensures reproducibility of results, and `n_init=10` runs the algorithm multiple times with different centroid seeds and picks the best result.
    *   The model is `fit` to the TF-IDF matrix `X`, and the resulting cluster assignments for each document are stored in `labels`.

3.  **Dimensionality Reduction (PCA)**:
    *   Since the TF-IDF vectors are typically high-dimensional, Principal Component Analysis (PCA) is applied to reduce these dimensions to `n_components=2`. This transformation allows the data to be easily plotted on a 2D scatter plot while retaining as much variance as possible. `random_state=42` ensures reproducibility.
    *   `X_reduced` holds the 2D coordinates for each document.

4.  **Plotting**:
    *   `matplotlib.pyplot` is used to create a scatter plot.
    *   Documents are plotted based on their `X_reduced` coordinates. Each document is colored according to its `labels` (cluster assignment), making it easy to visually distinguish the clusters.
    *   The plot includes a title, axis labels, a legend, and a grid for better readability.

5.  **Printing Cluster Assignments**:
    *   Finally, the script iterates through the original `documents` and their corresponding `labels` to print a clear list showing which cluster each document belongs to.

## License

This project is open-source and available under the [License Name] License. Please see the LICENSE file for more details.