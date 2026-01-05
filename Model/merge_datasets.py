import pandas as pd

# Load datasets from data folder
movies = pd.read_csv("../data/movies.csv")
links = pd.read_csv("../data/links.csv")
tmdb = pd.read_csv("../data/tmdb_movies_1990_2024.csv")

# Standardize column name
tmdb.rename(columns={"tmdb_id": "tmdbId"}, inplace=True)

# Clean tmdbId
links = links.dropna(subset=["tmdbId"])
links["tmdbId"] = links["tmdbId"].astype(int)
tmdb["tmdbId"] = tmdb["tmdbId"].astype(int)

# Merge movies + links
movies_links = movies.merge(
    links,
    on="movieId",
    how="left"
)

# Merge with TMDB metadata
final_movies_metadata = movies_links.merge(
    tmdb,
    on="tmdbId",
    how="left"
)


# Save final dataset (you can save in data or Model)
final_movies_metadata.to_csv("../data/final_movies_metadata.csv", index=False)

print(" Movies metadata created successfully!")
print("Rows:", final_movies_metadata.shape[0])
print("Columns:", final_movies_metadata.shape[1])