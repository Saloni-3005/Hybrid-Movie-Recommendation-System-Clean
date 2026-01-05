import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1: LOAD RATINGS
ratings = pd.read_csv("../data/ratings.csv")
ratings = ratings[["userId", "movieId", "rating"]]

# Map movieId and userId to continuous indices
movie_ids = ratings['movieId'].unique()
user_ids = ratings['userId'].unique()

movie_map = {id:i for i,id in enumerate(movie_ids)}
user_map = {id:i for i,id in enumerate(user_ids)}

ratings['movie_idx'] = ratings['movieId'].map(movie_map)
ratings['user_idx'] = ratings['userId'].map(user_map)


# STEP 2: CREATE SPARSE MATRIX
# rows = movies, cols = users, data = ratings
user_item_sparse = csr_matrix(
    (ratings['rating'], (ratings['movie_idx'], ratings['user_idx']))
)

# -----------------------------
# STEP 3: COMPUTE ITEM-ITEM SIMILARITY
# -----------------------------
# Use sparse similarity (memory-efficient)
item_sim = cosine_similarity(user_item_sparse, dense_output=False)

# STEP 4: SAVE MODELS
with open("../data/item_similarity.pkl", "wb") as f:
    pickle.dump(item_sim, f)

with open("../data/user_item_sparse.pkl", "wb") as f:
    pickle.dump(user_item_sparse, f)

with open("../data/movie_map.pkl", "wb") as f:
    pickle.dump(movie_map, f)

print("Sparse collaborative filtering model created successfully!")