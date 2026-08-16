# 🎬 Movie Recommendation System

A content-based Movie Recommendation System that recommends movies similar to a selected title. The application leverages metadata from TMDB, calculates cosine similarity using text vectorization, and features a responsive, dark-themed Streamlit web interface with dynamic poster fetching.

---

## ✨ Features

- **Content-Based Filtering**: Analyzes movie descriptions, genres, keywords, cast, and crew to find similar movies.
- **Cosine Similarity Matrix**: Recommends the top 10 closest movies based on the cosine distance of their vectorized tags.
- **Dynamic Poster Fetching**: Uses the TMDB API to retrieve high-quality posters in parallel using Python's `ThreadPoolExecutor`.
- **Advanced Caching**: Keeps recommendations fast by caching successful API requests while handling transient network drops gracefully.
- **Premium UI/UX**: Custom dark-themed layout built with Streamlit, optimized for readability and aesthetics.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
Since the large similarity matrix (`movie_data.pkl`) is tracked via **Git LFS**, ensure you have [Git LFS](https://git-lfs.com/) installed before cloning:
```bash
git lfs install
git clone https://github.com/Anandakrishnna/MovieRecommendationSystem.git
cd MovieRecommendationSystem
```

### 2. Set Up Environment & Dependencies
Create a virtual environment and install the required libraries:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add TMDB API Key
This project requires a TMDB API Key to fetch movie posters.
1. Sign up on [The Movie Database (TMDB)](https://www.themoviedb.org/) and generate an API key.
2. In the root directory, create a `.streamlit` folder and add a `secrets.toml` file:
   ```toml
   # .streamlit/secrets.toml
   TMDB_API_KEY = "your_tmdb_api_key_here"
   ```

---

## 🚀 Running the App

Start the Streamlit application with the following command:
```bash
streamlit run app.py
```

Access the application in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

- `app.py`: Streamlit application code containing UI definition, poster-fetching, and recommendation logic.
- `Movie_Recommendation_System.ipynb`: Jupyter notebook containing the raw data cleaning, vectorization (scikit-learn), and model generation steps.
- `movie_data.pkl`: Processed DataFrame and Cosine Similarity matrix (tracked with Git LFS).
- `requirements.txt`: Required dependencies (`streamlit`, `pandas`, `requests`, `scikit-learn`).
