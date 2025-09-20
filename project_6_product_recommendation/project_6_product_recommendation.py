```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Step 1: Create User-Product Rating Matrix
# -----------------------------
data = {
    'Product A': [5, 4, 0, 0, 1, 2, 0],
    'Product B': [3, 0, 0, 5, 1, 0, 4],
    'Product C': [4, 3, 0, 0, 1, 2, 0],
    'Product D': [0, 0, 5, 4, 0, 0, 3],
    'Product E': [0, 2, 4, 0, 0, 1, 0],
    'Product F': [1, 0, 2, 0, 5, 0, 4],
    'Product G': [0, 3, 0, 4, 2, 0, 1],
    'Product H': [5, 0, 1, 0, 0, 4, 0]
}

user_ids = ['User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'User 6', 'User 7']
ratings_df = pd.DataFrame(data, index=user_ids)

print("User-Product Ratings:\n")
print(ratings_df)

# -----------------------------
# Step 2: Create Product-User Matrix
# -----------------------------
product_user_matrix = ratings_df.T

# -----------------------------
# Step 3: Compute Cosine Similarity
# -----------------------------
similarity_matrix = pd.DataFrame(
    cosine_similarity(product_user_matrix),
    index=product_user_matrix.index,
    columns=product_user_matrix.index
)

print("\nProduct Similarity Matrix:\n")
print(similarity_matrix.round(2))

# -----------------------------
# Step 4: Recommendation Function
# -----------------------------
def recommend_products(product_name, similarity_matrix, top_n=3):
    """Recommend top-N similar products for a given product."""
    if product_name not in similarity_matrix:
        return f"❌ Product '{product_name}' not found."

    # Sort similarity scores (excluding the product itself)
    sorted_similar_products = similarity_matrix[product_name].sort_values(ascending=False)
    recommendations = sorted_similar_products[1:top_n+1]

    return recommendations

# -----------------------------
# Step 5: Example Recommendation
# -----------------------------
print("\nRecommended products for 'Product A':\n")
print(recommend_products("Product A", similarity_matrix, top_n=3))
```