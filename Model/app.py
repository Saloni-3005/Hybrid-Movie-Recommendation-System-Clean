# app.py
import streamlit as st
import sqlite3
import hashlib
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

# -----------------------------
# DATABASE SETUP (SQLite)
# -----------------------------
DB_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

def get_db():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def create_users_table():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        conn.close()
        return True
    except:
        return False

def authenticate_user(username, password):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    return user

create_users_table()

# -----------------------------
# SESSION STATE
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "search_history" not in st.session_state:
    st.session_state.search_history = []


# ------------------------------------------------
# PATHS
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))  

# ------------------------------------------------
# LOAD DATA (MEMORY SAFE)
# ------------------------------------------------
@st.cache_data
def load_data():
    movies_csv = os.path.join(DATA_DIR, "final_movies_metadata.csv")
    if not os.path.exists(movies_csv):
        st.error(f"CSV file not found at {movies_csv}")
        st.stop()
    movies = pd.read_csv(movies_csv)

    movies["title"] = movies["title_x"]
    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    movies["content"] = movies["genres"] + " " + movies["overview"]

    user_item_file = os.path.join(DATA_DIR, "user_item_sparse.pkl")
    movie_map_file = os.path.join(DATA_DIR, "movie_map.pkl")
    if not os.path.exists(user_item_file) or not os.path.exists(movie_map_file):
        st.error("Required pickle files not found in data folder!")
        st.stop()

    user_item_sparse = pickle.load(open(user_item_file, "rb"))
    movie_map = pickle.load(open(movie_map_file, "rb"))

    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = tfidf.fit_transform(movies["content"])

    return movies, user_item_sparse, movie_map, tfidf_matrix

movies, user_item_sparse, movie_map, tfidf_matrix = load_data()

# -----------------------------
# HYBRID RECOMMENDER
# -----------------------------
def hybrid_recommend(title, top_n=10):
    if title not in movies["title"].values:
        return pd.DataFrame()
    idx = movies[movies["title"] == title].index[0]
    movie_id = movies.iloc[idx]["movieId"]

    # Content-based
    content_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    content_idx = content_sim.argsort()[::-1][1:top_n*2+1]
    content_df = movies.iloc[content_idx]

    # Collaborative
    if movie_id in movie_map:
        movie_idx = movie_map[movie_id]
        movie_vector = user_item_sparse[movie_idx]
        sim = cosine_similarity(movie_vector, user_item_sparse)[0]
        top_indices = sim.argsort()[::-1][1:top_n*2+1]
        inv_map = {v: k for k, v in movie_map.items()}
        movie_ids = [inv_map[i] for i in top_indices]
        collab_df = movies[movies["movieId"].isin(movie_ids)]
    else:
        collab_df = pd.DataFrame()

    return pd.concat([content_df, collab_df]).drop_duplicates("movieId").head(top_n)

# -----------------------------
# UI COMPONENTS
# -----------------------------
def movie_cards(df, columns_per_row=5):
    for i in range(0, len(df), columns_per_row):
        cols = st.columns(columns_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(df):
                break
            m = df.iloc[i + j]
            with col:
                if pd.notna(m.get("poster_path")):
                    st.image(
                        "https://image.tmdb.org/t/p/w300" + m["poster_path"],
                        use_container_width=True
                    )
                st.markdown(f"**🎬 {m['title']}**")
                st.markdown(f"⭐ {m.get('vote_average', 'N/A')}")
                with st.expander("Overview"):
                    st.write(m.get("overview", "No description available"))

# -----------------------------
# PAGES
# -----------------------------
def home_page():
    st.title("🎥 Welcome to Hybrid Movie Recommender")
    st.markdown("Netflix/Hotstar style UI with login authentication")
    col1, col2 = st.columns(2)
    
    st.markdown("---")
    st.markdown("### Popular Movies")
    movie_cards(movies.head(7))

def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = authenticate_user(username, password)
        if user:
            st.success("Login successful")
            st.session_state.logged_in = True
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Invalid username or password")

def register_page():
    st.title("📝 Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if password != confirm:
            st.error("Passwords do not match")
        elif len(password) < 4:
            st.error("Password too short")
        elif register_user(username, password):
            st.success("Registration successful! Please login.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Username already exists")

def dashboard_page():
    st.title("🎬 Movie Recommendation Dashboard")
    
    # Search
    search_term = st.text_input("Search a movie...")
    suggestions = movies[movies["title"].str.lower().str.contains(search_term.lower())]["title"].tolist() if search_term else []
    movie = st.selectbox("Select a movie", suggestions if suggestions else movies["title"].unique())

    # Store search history
    if search_term and search_term not in st.session_state.search_history:
        st.session_state.search_history.insert(0, search_term)  # newest first
        st.session_state.search_history = st.session_state.search_history[:10]  # keep last 10 searches

    # Show search history in sidebar
    with st.sidebar.expander("🔍 Search History"):
        if st.session_state.search_history:
            for past in st.session_state.search_history:
                if st.button(past):
                    st.session_state.page = "dashboard"
                    st.experimental_rerun()
        else:
            st.write("No search history yet.")

    # View details button
    if st.button("View Details"):
        movie_id = movies[movies["title"] == movie]["movieId"].values[0]
        st.session_state.selected_movie = movie_id
        st.session_state.page = "movie_detail"
        st.rerun()
    
    # Recommendations
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    if st.button("Recommend"):
        recs = hybrid_recommend(movie, top_n)
        movie_cards(recs)

def logout():
    st.session_state.logged_in = False
    st.session_state.page = "home"
    st.session_state.selected_movie = None
    st.rerun()

def movie_detail_page():
    movie_id = st.session_state.get("selected_movie")
    if not movie_id:
        st.warning("No movie selected.")
        st.session_state.page = "dashboard"
        st.rerun()
        return

    movie = movies[movies["movieId"] == movie_id].iloc[0]

    st.title(movie["title"])
    col1, col2 = st.columns([1, 2])
    with col1:
        if pd.notna(movie.get("poster_path")):
            st.image("https://image.tmdb.org/t/p/w300" + movie["poster_path"], use_container_width=True)
    with col2:
        st.markdown(f"**Release Date:** {movie.get('release_date', 'N/A')}")
        st.markdown(f"**Genres:** {movie.get('genres', 'N/A')}")
        st.markdown(f"**Rating:** ⭐ {movie.get('vote_average', 'N/A')}")
        st.markdown("### Overview")
        st.write(movie.get("overview", "No description available"))

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
pages = {
    "🏠 Home": "home",
    "🔐 Login": "login",
    "📝 Register": "register",
    "🎬 Dashboard": "dashboard"
}

if st.session_state.get("selected_movie"):
    pages["📄 Movie Detail"] = "movie_detail"

st.sidebar.title("📂 Navigation")
selected_page = st.sidebar.radio("Go to", list(pages.keys()), index=list(pages.values()).index(st.session_state.get("page", "home")))
current_page = pages[selected_page]

if st.session_state.get("logged_in") and current_page != "login":
    if st.sidebar.button("🚪 Logout"):
        logout()

# -----------------------------
# ROUTER
# -----------------------------
if current_page == "home":
    home_page()
elif current_page == "login":
    login_page()
elif current_page == "register":
    register_page()
elif current_page == "dashboard":
    if st.session_state.logged_in:
        dashboard_page()
    else:
        st.session_state.page = "login"
        st.rerun()
elif current_page == "movie_detail":
    movie_detail_page()
