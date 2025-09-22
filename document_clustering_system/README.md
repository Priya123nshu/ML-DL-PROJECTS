# Document Clustering System

## Overview
This Python script demonstrates a basic document clustering system. It takes a predefined set of text documents, processes them using TF-IDF vectorization, applies the K-Means clustering algorithm to group them into distinct clusters, and then visualizes these clusters in a 2D space using Principal Component Analysis (PCA). Finally, it prints each document along with its assigned cluster.

## Installation
The script requires the following Python libraries. You can install them using pip:

```bash
pip install scikit-learn matplotlib
```

## Usage
To run this script, save the code as a Python file (e.g., `cluster_documents.py`) and execute it from your terminal:

```bash
python cluster_documents.py
```

The script will:
1. Display a scatter plot visualizing the clustered documents.
2. Print the cluster assignments for each document to the console.

You can modify the `documents` list in the script to cluster your own set of texts.

## Inputs & Outputs

### Inputs
The script uses a hardcoded list of strings called `documents` as its input. Each string in the list represents a document to be clustered.

Example:
```python
documents = [
    "Apple released a new iPhone today.",
    "The stock market saw major gains.",
    # ... more documents
]
```

### Outputs
1.  **Graphical Output**: A Matplotlib window displaying a 2D scatter plot titled "Document Clustering with K-Means". Each point on the plot represents a document, and points of the same color belong to the same cluster. The axes represent the first two principal components.

    ```
    (A plot window will appear showing clustered points.)
    ```

2.  **Console Output**: A list of the original documents, each prefixed with its assigned cluster number.

    ```
    Cluster 1: Apple released a new iPhone today.
    Cluster 2: The stock market saw major gains.
    Cluster 1: Google announced a new Android update.
    # ... and so on
    ```

## Explanation
The script performs document clustering through a series of well-defined steps:

1.  **TF-IDF Vectorization**:
    - The `TfidfVectorizer` from `sklearn.feature_extraction.text` is used to convert the raw text documents into numerical feature vectors.
    - TF-IDF (Term Frequency-Inverse Document Frequency) assigns weights to words based on their frequency in a document and their rarity across all documents, making important words stand out.
    - English stop words (common words like "the", "is", "and") are removed during this process to focus on more meaningful terms.

2.  **K-Means Clustering**:
    - The `KMeans` algorithm from `sklearn.cluster` is applied to the TF-IDF vectors.
    - The script is configured to find `n_clusters = 3` distinct clusters.
    - `random_state=42` ensures reproducibility of the results.
    - `n_init=10` specifies that the K-Means algorithm will be run 10 times with different centroid seeds, and the best result (in terms of inertia) will be chosen, making the clustering more robust.
    - The `labels_` attribute of the fitted model stores the cluster assignment for each document.

3.  **Dimensionality Reduction (PCA)**:
    - `PCA` (Principal Component Analysis) from `sklearn.decomposition` is used to reduce the high dimensionality of the TF-IDF vectors (which can have thousands of features) down to `n_components=2`.
    - This reduction is essential for visualizing the clusters on a 2D scatter plot while retaining as much variance as possible from the original data.
    - `random_state=42` ensures reproducibility.

4.  **Visualization**:
    - `matplotlib.pyplot` is used to create a scatter plot.
    - Documents belonging to the same cluster are plotted with the same color, allowing for a visual inspection of the clustering results.
    - The plot includes a title, axis labels, a legend for clusters, and a grid for better readability.

5.  **Cluster Output**:
    - Finally, the script iterates through the original documents and their corresponding cluster labels, printing them to the console. This provides a clear, textual breakdown of which documents were grouped together.

## License
This project is licensed under the [Your Chosen License] - see the LICENSE.md file for details.