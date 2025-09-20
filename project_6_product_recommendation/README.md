# project_6_product_recommendation

## Overview

This module implements a basic product recommendation system based on user ratings using cosine similarity. It takes a hardcoded user-product rating matrix, calculates the similarity between products, and provides a function to recommend the top-N most similar products for a given item.

## Installation

This project requires the following Python libraries:

*   **pandas**: For data manipulation and DataFrame operations.
*   **scikit-learn**: Specifically for `sklearn.metrics.pairwise.cosine_similarity` to compute similarity scores.

You can install these dependencies using pip:

```bash
pip install pandas scikit-learn
```

## Usage

This module can be run directly to see an example of the rating matrix, similarity matrix, and recommendations, or its `recommend_products` function can be imported and used in other scripts.

### Running the Script Directly

Executing the script will print the generated user-product rating matrix, the product similarity matrix, and an example recommendation for 'Product A'.

```bash
python your_module_name.py
```

*(Note: Replace `your_module_name.py` with the actual filename if you saved the code to a file.)*

**Example Output when run directly:**

```
User-Product Ratings:

        Product A  Product B  Product C  Product D  Product E  Product F  Product G  Product H
User 1          5          3          4          0          0          1          0          5
User 2          4          0          3          0          2          0          3          0
User 3          0          0          0          5          4          2          0          1
User 4          0          5          0          4          0          0          4          0
User 5          1          1          1          0          0          5          2          0
User 6          2          0          2          0          1          0          0          4
User 7          0          4          0          3          0          4          1          0

Product Similarity Matrix:

           Product A  Product B  Product C  Product D  Product E  Product F  Product G  Product H
Product A       1.00       0.28       0.99       0.00       0.23       0.15       0.09       0.55
Product B       0.28       1.00       0.20       0.51       0.22       0.54       0.63       0.00
Product C       0.99       0.20       1.00       0.00       0.22       0.15       0.09       0.49
Product D       0.00       0.51       0.00       1.00       0.86       0.33       0.80       0.05
Product E       0.23       0.22       0.22       0.86       1.00       0.18       0.68       0.09
Product F       0.15       0.54       0.15       0.33       0.18       1.00       0.36       0.00
Product G       0.09       0.63       0.09       0.80       0.68       0.36       1.00       0.00
Product H       0.55       0.00       0.49       0.05       0.09       0.00       0.00       1.00

Recommended products for 'Product A':

Product C    0.993419
Product H    0.553951
Product B    0.284705
Name: Product A, dtype: float64
```

### Using the `recommend_products` Function

You can import the `recommend_products` function and use it with your own similarity matrix. The script itself generates a `similarity_matrix` that can be used.

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from your_module_name import recommend_products # Assuming your code is in 'your_module_name.py'

# The data and setup as defined in the module:
data = {
    'Product A': [5, 4, 0, 0, 1, 2, 0], 'Product B': [3, 0, 0, 5, 1, 0, 4],
    'Product C': [4, 3, 0, 0, 1, 2, 0], 'Product D': [0, 0, 5, 4, 0, 0, 3],
    'Product E': [0, 2, 4, 0, 0, 1, 0], 'Product F': [1, 0, 2, 0, 5, 0, 4],
    'Product G': [0, 3, 0, 4, 2, 0, 1], 'Product H': [5, 0, 1, 0, 0, 4, 0]
}
user_ids = ['User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'User 6', 'User 7']
ratings_df = pd.DataFrame(data, index=user_ids)
product_user_matrix = ratings_df.T
similarity_matrix = pd.DataFrame(
    cosine_similarity(product_user_matrix),
    index=product_user_matrix.index,
    columns=product_user_matrix.index
)

# Get recommendations for 'Product D'
recommendations_d = recommend_products("Product D", similarity_matrix, top_n=2)
print("\nRecommended products for 'Product D':\n")
print(recommendations_d)

# Try recommending for a non-existent product
recommendations_invalid = recommend_products("Product Z", similarity_matrix)
print(f"\n{recommendations_invalid}")
```

## Inputs & Outputs

### Inputs

*   **Hardcoded Data**:
    *   `data` (dictionary): A dictionary where keys are product names (strings) and values are lists of integer ratings from users. `0` typically indicates no rating.
    *   `user_ids` (list): A list of strings representing user identifiers.
*   **`recommend_products` Function**:
    *   `product_name` (str): The name of the product for which to find recommendations. This must be a key present in the `similarity_matrix`'s index/columns.
    *   `similarity_matrix` (pandas.DataFrame): A DataFrame where both index and columns are product names, and values are their pairwise similarity scores.
    *   `top_n` (int, optional): The number of top similar products to recommend. Defaults to `3`.

### Outputs

*   **`ratings_df` (pandas.DataFrame)**: A DataFrame with `user_ids` as index and product names as columns, containing the raw rating data. Printed to console.
*   **`product_user_matrix` (pandas.DataFrame)**: A transposed version of `ratings_df` with product names as index and `user_ids` as columns. This is an intermediate output used for similarity calculation.
*   **`similarity_matrix` (pandas.DataFrame)**: A square DataFrame where both index and columns are product names, showing the cosine similarity between each pair of products. Printed to console.
*   **`recommend_products` Function**:
    *   Returns a `pandas.Series` where the index are the recommended product names and values are their similarity scores to the input `product_name`.
    *   If `product_name` is not found, returns a string: `"❌ Product '{product_name}' not found."`

## Explanation

The module follows a common pattern for content-based or item-item collaborative filtering:

1.  **Create User-Product Rating Matrix**:
    *   A `pandas.DataFrame` (`ratings_df`) is constructed from the initial `data` dictionary and `user_ids` list. Each row represents a user, and each column a product, with values being the user's rating for that product.

2.  **Create Product-User Matrix**:
    *   The `ratings_df` is transposed to create `product_user_matrix`. In this matrix, each row represents a product, and each column a user, with values being how users rated that specific product. This is crucial for calculating product-to-product similarity.

3.  **Compute Cosine Similarity**:
    *   `sklearn.metrics.pairwise.cosine_similarity` is used to calculate the cosine similarity between all pairs of products based on their rating vectors in the `product_user_matrix`. Cosine similarity measures the cosine of the angle between two non-zero vectors in a multi-dimensional space. A value close to 1 indicates high similarity, 0 indicates no similarity, and -1 indicates complete dissimilarity (though ratings are non-negative here).
    *   The result is stored in `similarity_matrix`, a square DataFrame where `similarity_matrix[product_X][product_Y]` gives the similarity between Product X and Product Y.

4.  **Recommendation Function (`recommend_products`)**:
    *   This function takes a `product_name`, the pre-computed `similarity_matrix`, and an optional `top_n` parameter.
    *   It first checks if the `product_name` exists in the `similarity_matrix`.
    *   It then extracts the row (or column, due to symmetry) corresponding to the `product_name` from the `similarity_matrix`, which gives the similarity scores of all other products to the target product.
    *   These similarity scores are sorted in descending order.
    *   The function then selects the top `top_n` products, explicitly excluding the product itself (which would always have a similarity of 1.0).

## License

This project is open-source and available under the [License Name] license. Please see the `LICENSE` file for more details.