import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load merged movies metadata
movies = pd.read_csv("../data/final_movies_metadata.csv")

# Use TMDB title (title_y)
movies["title"] = movies["title_y"]

# Keep only required columns
movies = movies[["movieId", "title", "genres", "overview"]]

# Handle missing values
movies["genres"] = movies["genres"].fillna("")
movies["overview"] = movies["overview"].fillna("")

# Combine text features for content-based recommendation
movies["content"] = movies["genres"] + " " + movies["overview"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies["content"])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save models for later use
with open("../data/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("../data/cosine_sim.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)

# Save movies index for lookup
movies.to_pickle("../data/movies_index.pkl")

print("Content-based recommender model created successfully!")