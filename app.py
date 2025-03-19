import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, hamming
from scipy.stats import pearsonr
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


scaler = StandardScaler()
normalized_ratings = scaler.fit_transform(df)


cosine_sim = cosine_similarity(normalized_ratings)
cosine_df = pd.DataFrame(cosine_sim, index=df.index, columns=df.index)


def euclidean_similarity(matrix):
    n = matrix.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = euclidean(matrix[i], matrix[j])
    return pd.DataFrame(dist_matrix, index=df.index, columns=df.index)

euclidean_df = euclidean_similarity(normalized_ratings)


def pearson_similarity(matrix):
    n = matrix.shape[0]
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i][j], _ = pearsonr(matrix[i], matrix[j])
    return pd.DataFrame(corr_matrix, index=df.index, columns=df.index)

pearson_df = pearson_similarity(normalized_ratings)


def hamming_similarity(matrix):
    n = matrix.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = hamming(matrix[i], matrix[j])
    return pd.DataFrame(dist_matrix, index=df.index, columns=df.index)

hamming_df = hamming_similarity(normalized_ratings)


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


st.title("Behind the scenes of an Recommendation System Model")


# Show User-User Cosine Similarity
st.subheader("User Similarity (Cosine Similarity)")
st.dataframe(cosine_df)
fig, ax = plt.subplots()
sns.heatmap(cosine_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)
st.write("**Explanation:** Cosine similarity measures the angle between two vectors. Higher values mean users are more similar.")

# Show User-User Euclidean Distance
st.subheader("User Similarity (Euclidean Distance)")
st.dataframe(euclidean_df)
fig, ax = plt.subplots()
sns.heatmap(euclidean_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)
st.write("**Explanation:** Euclidean distance measures the straight-line distance between users' rating vectors. Lower values mean more similarity.")

# Show User-User Pearson Correlation
st.subheader("User Similarity (Pearson Correlation)")
st.dataframe(pearson_df)
fig, ax = plt.subplots()
sns.heatmap(pearson_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)
st.write("**Explanation:** Pearson correlation measures the linear relationship between users' ratings. Higher values indicate stronger correlation.")

# Show User-User Hamming Distance
st.subheader("User Similarity (Hamming Distance)")
st.dataframe(hamming_df)
fig, ax = plt.subplots()
sns.heatmap(hamming_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)
st.write("**Explanation:** Hamming distance counts the number of differences between users' rating patterns. Lower values mean users are more similar.")

# Show Movie Similarity (Content-based Filtering)
st.subheader("Movie Similarity (Cosine Similarity on Content)")
st.dataframe(movie_sim_df)
fig, ax = plt.subplots()
sns.heatmap(movie_sim_df, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)
st.write("**Explanation:** Cosine similarity on TF-IDF vectors determines content similarity between movies.")

# User Input for Recommendation
selected_user = st.selectbox("Select a user to recommend movies for:", df.index)
st.subheader(f"Recommended Movies for {selected_user}")
user_sim_scores = cosine_df[selected_user].drop(selected_user).sort_values(ascending=False)
most_similar_user = user_sim_scores.idxmax()
st.write(f"Most similar user: {most_similar_user}")
st.write("Movies rated highest by similar user:")
st.dataframe(df.loc[most_similar_user].sort_values(ascending=False))


