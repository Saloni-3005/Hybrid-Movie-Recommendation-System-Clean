import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# STEP 1: LOAD DATA
movies = pd.read_csv("../data/final_movies_metadata.csv")
ratings = pd.read_csv("../data/ratings.csv")

# Fix title column
movies['title'] = movies['title_x']  # choose one title column
movies['overview'] = movies['overview'].fillna('')
movies['content'] = movies['genres'].fillna('') + " " + movies['overview']

# Load sparse user-item matrix and movie map
with open("../data/user_item_sparse.pkl", "rb") as f:
    user_item_sparse = pickle.load(f)

with open("../data/movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)


# STEP 2: TF-IDF CONTENT-BASED MODEL

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies['content'])


# STEP 3: HELPER FUNCTIONS

def get_movie_index(movie_title):
    df = movies[movies['title'].str.lower() == movie_title.lower()]
    if df.empty:
        return None
    return df.index[0]

# Content-based recommendations
def content_recommendations(movie_idx, top_n=10):
    sim_scores = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix)[0]
    top_indices = sim_scores.argsort()[::-1][1:top_n+1]
    return movies.iloc[top_indices][['movieId','title']]

# Collaborative filtering recommendations (memory-efficient)
def collaborative_recommendations(movie_id, top_n=10):
    if movie_id not in movie_map:
        return pd.DataFrame(columns=['movieId','title'])

    movie_idx = movie_map[movie_id]
    movie_vec = user_item_sparse[movie_idx]

    # Compute similarity only for this movie (avoid full dense matrix)
    sim = cosine_similarity(movie_vec, user_item_sparse)[0]  # memory-efficient
    top_indices = sim.argsort()[::-1][1:top_n+1]

    # Map back to movieId
    inv_movie_map = {v:k for k,v in movie_map.items()}
    movie_ids = [inv_movie_map[i] for i in top_indices]

    return movies[movies['movieId'].isin(movie_ids)][['movieId','title']]

# Hybrid recommendations
def hybrid_recommendations(movie_title, top_n=10, alpha=0.5):
    movie_idx = get_movie_index(movie_title)
    if movie_idx is None:
        return f"Movie '{movie_title}' not found."

    movie_id = movies.iloc[movie_idx]['movieId']

    # Content-based top N
    content_df = content_recommendations(movie_idx, top_n*2)

    # Collaborative top N
    collab_df = collaborative_recommendations(movie_id, top_n*2)

    # Merge content + collaborative (simple deduplication)
    merged = pd.concat([content_df, collab_df]).drop_duplicates('movieId').head(top_n)
    return merged.reset_index(drop=True)


# STEP 4: TEST THE RECOMMENDER

if __name__ == "__main__":
    test_movie = "Toy Story (1995)"
    print(f"Input Movie: {test_movie}\n")
    print("Top 10 Recommendations:\n")
    print(hybrid_recommendations(test_movie, top_n=10))