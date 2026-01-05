import requests
import pandas as pd
import time

API_KEY = "315f1684f0915148d6955b57980b8f95"
BASE_URL = "https://api.themoviedb.org/3/discover/movie"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

movies = []

for year in range(1990, 2025):
    print(f"Fetching year {year}")

    for page in range(1, 3):   # 2 pages only (safe)
        params = {
            "api_key": API_KEY,
            "primary_release_year": year,
            "sort_by": "popularity.desc",
            "page": page
        }

        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=headers,
                timeout=10
            )

            if response.status_code != 200:
                print("Skipped:", response.status_code)
                time.sleep(5)
                continue

            data = response.json()

            for movie in data.get("results", []):
                movies.append({
                    "tmdb_id": movie.get("id"),
                    "title": movie.get("title"),
                    "release_date": movie.get("release_date"),
                    "overview": movie.get("overview"),
                    "vote_average": movie.get("vote_average"),
                    "vote_count": movie.get("vote_count"),
                    "popularity": movie.get("popularity")
                })

            time.sleep(1.5)   # IMPORTANT: slow & stable

        except Exception as e:
            print("Error:", e)
            time.sleep(10)
            continue

df = pd.DataFrame(movies)
df.to_csv("tmdb_movies_1990_2024.csv", index=False)

print("TMDB movie data saved successfully")