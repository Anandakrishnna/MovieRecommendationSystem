import os
import pickle
import requests
import streamlit as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-quality fallback poster URL when API poster is unavailable or unconfigured
FALLBACK_POSTER_URL = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500&auto=format&fit=crop"


@st.cache_resource
def load_data():
    """Load preprocessed movie DataFrame and Cosine Similarity matrix."""
    file_path = "movie_data.pkl"
    if not os.path.exists(file_path):
        st.error(f"Error: Processed data file '{file_path}' not found. Please ensure it is present in the app directory.")
        st.stop()
    try:
        with open(file_path, "rb") as file:
            movies, cosine_sim = pickle.load(file)
        return movies, cosine_sim
    except Exception as e:
        st.error(f"Failed to load movie data: {e}")
        st.stop()


movies, cosine_sim = load_data()


def get_recommendations(title, cosine_sim_matrix=cosine_sim):
    """Retrieve top 10 recommended movies based on cosine similarity."""
    matching_movies = movies[movies["title"] == title]
    if matching_movies.empty:
        return pd.DataFrame(columns=["title", "movie_id"])

    idx = matching_movies.index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    # Sort movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get top 10 most similar movies (excluding the searched movie itself at index 0)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies[["title", "movie_id"]].iloc[movie_indices]


@st.cache_data(show_spinner=False)
def fetch_poster_url(movie_id):
    """Fetch movie poster URL from TMDB API with caching and retry logic."""
    try:
        api_key = st.secrets.get("TMDB_API_KEY")
    except Exception:
        api_key = None

    if not api_key or api_key == "your_tmdb_api_key_here":
        return FALLBACK_POSTER_URL

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get("poster_path")
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
                break
        except requests.exceptions.RequestException:
            continue
    return FALLBACK_POSTER_URL


def fetch_poster(movie_id):
    """Wrapper function to fetch poster safely."""
    try:
        return fetch_poster_url(movie_id)
    except Exception:
        return FALLBACK_POSTER_URL


# Custom Dark Cinema Theme Styling
st.markdown("""
    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header & Navigation Bar Reset */
    header[data-testid="stHeader"] {
        background-color: #0e1117 !important;
    }
    [data-testid="stToolbar"] {
        background-color: #0e1117 !important;
    }
    [data-testid="stDecoration"] {
        background-image: none !important;
        background-color: #0e1117 !important;
    }
    
    /* Main Layout Container */
    .main .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        margin: 0 auto;
    }
    
    /* Title & Description Header */
    .title-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .title-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.4rem !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    .title-header p {
        color: #9da4b0 !important;
        font-size: 1rem !important;
        max-width: 700px;
        margin: 0 auto 1.5rem auto;
    }
    
    /* Badges */
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .badge {
        background-color: #1e2330;
        color: #e2e8f0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #2d3748;
    }
    
    /* Selectbox Styling */
    .stSelectbox label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }
    [data-testid="stSelectbox"] > div > div {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    [data-testid="stSelectbox"] * {
        color: #ffffff !important;
    }
    
    /* Button Styling */
    .stButton {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #e50914 0%, #b81d24 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        padding: 10px 32px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(229, 9, 20, 0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #f40612 0%, #c91820 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6);
        color: #ffffff !important;
    }
    
    /* Movie Poster Cards */
    .movie-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 8px;
        text-align: center;
        border: 1px solid #21262d;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        border-color: #30363d;
    }
    .movie-title {
        color: #f0f6fc;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
        line-height: 1.25;
        height: 2.5em;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    img {
        border-radius: 8px;
        object-fit: cover;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# Application Header
st.markdown("""
    <div class="title-header">
        <h1>🎬 Movie Recommendation System</h1>
        <p>Discover personalized movie recommendations powered by metadata similarity, TF-IDF text vectorization, and TMDB API integration.</p>
        <div class="badge-container">
            <span class="badge">Content-Based Filtering</span>
            <span class="badge">TF-IDF Vectorizer</span>
            <span class="badge">Cosine Similarity</span>
            <span class="badge">TMDB Integration</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# TMDB API Key Warning Check
try:
    tmdb_key = st.secrets.get("TMDB_API_KEY", "")
except Exception:
    tmdb_key = ""

if not tmdb_key or tmdb_key == "your_tmdb_api_key_here":
    st.info("ℹ️ **TMDB API Key missing or default**: Posters are using high-resolution fallback visuals. Configure `TMDB_API_KEY` in `.streamlit/secrets.toml` to fetch live movie posters.")

# Movie Selector
selected_movie = st.selectbox(
    "Choose a movie you like:",
    movies["title"].values,
    index=0
)

# Recommendation Trigger
if st.button("🚀 Get Recommendations"):
    with st.spinner("Finding recommendations and fetching posters..."):
        recommendations = get_recommendations(selected_movie)
        
        if recommendations.empty:
            st.warning("No recommendations found for the selected movie.")
        else:
            st.subheader(f"Top 10 movies similar to '{selected_movie}':")
            
            movie_ids = recommendations["movie_id"].tolist()
            titles = recommendations["title"].tolist()

            # Parallel poster fetching using ThreadPoolExecutor for low-latency loading
            with ThreadPoolExecutor(max_workers=5) as executor:
                posters = list(executor.map(fetch_poster, movie_ids))

            # Display recommendations in 2 rows of 5 cards
            for i in range(0, len(titles), 5):
                cols = st.columns(5)
                for col, j in zip(cols, range(i, min(i + 5, len(titles)))):
                    with col:
                        st.markdown(f"""
                            <div class="movie-card">
                                <img src="{posters[j]}" alt="{titles[j]}"/>
                                <div class="movie-title">{titles[j]}</div>
                            </div>
                        """, unsafe_allow_html=True)
