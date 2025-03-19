import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample Movie Ratings Dataset
data = {
    "User": ["A", "B", "C", "D"],
    "Movie1": [5, 3, 4, 2],
    "Movie2": [4, 2, 5, 3],
    "Movie3": [2, 5, 3, 4],
    "Movie4": [3, 4, 2, 5],
}
df = pd.DataFrame(data).set_index("User")

# Normalize the ratings
scaler = StandardScaler()
normalized_ratings = scaler.fit_transform(df)

# Compute Cosine Similarity
cosine_sim = cosine_similarity(normalized_ratings)
cosine_df = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)

# Compute Euclidean Distance
def euclidean_similarity(matrix):
    n = matrix.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = euclidean(matrix[i], matrix[j])
    return pd.DataFrame(dist_matrix, index=df.index, columns=df.index)

euclidean_df = euclidean_similarity(normalized_ratings)

# Content-based Filtering (TF-IDF on Movie Descriptions)
movies = {
    "Movie1": "Action Adventure",
    "Movie2": "Romance Drama",
    "Movie3": "Sci-Fi Thriller",
    "Movie4": "Action Sci-Fi",
}
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies.values())
cosine_movie_sim = cosine_similarity(tfidf_matrix)
movie_sim_df = pd.DataFrame(cosine_movie_sim, index=movies.keys(), columns=movies.keys())

# Streamlit App UI
st.title("Behind the Scene tools used in RS")


# Show User-User Cosine Similarity
st.subheader("User Similarity (Cosine Similarity)")
st.dataframe(cosine_df)
fig, ax = plt.subplots()
sns.heatmap(cosine_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# Show User-User Euclidean Distance
st.subheader("User Similarity (Euclidean Distance)")
st.dataframe(euclidean_df)
fig, ax = plt.subplots()
sns.heatmap(euclidean_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# Show Movie Similarity (Content-based Filtering)
st.subheader("Movie Similarity (Cosine Similarity on Content)")
st.dataframe(movie_sim_df)
fig, ax = plt.subplots()
sns.heatmap(movie_sim_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# User Input for Recommendation
selected_user = st.selectbox("Select a user to recommend movies for:", df.index)
st.subheader(f"Recommended Movies for {selected_user}")
user_sim_scores = cosine_df[selected_user].drop(selected_user).sort_values(ascending=False)
most_similar_user = user_sim_scores.idxmax()
st.write(f"Most similar user: {most_similar_user}")
st.write("Movies rated highest by similar user:")
st.dataframe(df.loc[most_similar_user].sort_values(ascending=False))


