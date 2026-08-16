import streamlit as st
import pandas as pd
import requests
import pickle
from concurrent.futures import ThreadPoolExecutor

# Page config - MUST be first Streamlit command
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Load the processed data and similarity matrix
with open('movie_data.pkl', 'rb') as file:
    movies, cosine_sim = pickle.load(file)

# Function to get movie recommendations


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'movie_id']].iloc[movie_indices]

# Fetch movie poster from TMDB API (cached so it only fetches once per movie)


@st.cache_data
def fetch_poster_url(movie_id):
    api_key = st.secrets["TMDB_API_KEY"]
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    last_exception = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'poster_path' in data and data['poster_path']:
                return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
            else:
                raise ValueError("No poster path available")
        except (requests.exceptions.RequestException, ValueError) as e:
            last_exception = e
            continue
    raise last_exception if last_exception else Exception("Failed to fetch poster")


def fetch_poster(movie_id):
    try:
        return fetch_poster_url(movie_id)
    except Exception:
        # High quality cinema placeholder image
        return "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=500&auto=format&fit=crop"



# Custom CSS - dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    h1, .stApp h1, [data-testid="stMarkdownContainer"] h1 {
        color: #ffffff !important;
        font-weight: 800;
    }
    .stSelectbox label {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #1a1d24 !important;
        border: 1px solid #333 !important;
        border-radius: 12px !important;
    }
    [data-testid="stSelectbox"] * {
        color: #ffffff !important;
    }
    .stButton > button {
        background-color: transparent;
        color: #ff4b4b;
        border: 2px solid #ff4b4b;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #ff4b4b;
        color: white;
    }
    img {
        border-radius: 8px;
    }
    header[data-testid="stHeader"] {
    background-color: #0e1117 !important;
}
[data-testid="stToolbar"] {
    background-color: #0e1117 !important;
}
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
.main .block-container {
    max-width: 1000px;
    padding-top: 3rem;
    padding-left: 5rem;
    padding-right: 5rem;
    margin: 0 auto;
}
h1, .stApp h1, [data-testid="stMarkdownContainer"] h1 {
    color: #ffffff !important;
    font-weight: 800;
    font-size: 2rem !important;
}
[data-testid="stMarkdownContainer"] p {
    font-size: 0.85rem !important;
}
.stImage p {
    font-size: 0.8rem !important;
}
    </style>
""", unsafe_allow_html=True)


# Streamlit UI
st.title("Movie Recommendation System")

selected_movie = st.selectbox("Select a movie:", movies['title'].values)

if st.button('Recommend'):
    recommendations = get_recommendations(selected_movie)
    st.write("Top 10 recommended movies:")

    movie_ids = recommendations['movie_id'].tolist()
    titles = recommendations['title'].tolist()

    # Fetch all posters in parallel - faster loading
    with ThreadPoolExecutor(max_workers=4) as executor:
        posters = list(executor.map(fetch_poster, movie_ids))

    for i in range(0, 10, 5):
        cols = st.columns(5)
        for col, j in zip(cols, range(i, i+5)):
            if j < len(posters):
                with col:
                    st.image(posters[j], width=130)
                    st.write(titles[j])
