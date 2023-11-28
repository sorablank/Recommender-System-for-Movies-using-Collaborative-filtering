import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from surprise import SVD
from surprise import Dataset
from surprise.reader import Reader
from collections import defaultdict
import random

# Load recommendations.csv
recommendations = pd.read_csv('Dataset/recommendations.csv')

# Sample: Use 1% of the data for faster processing
recommendations_subset = recommendations.sample(frac=0.01, random_state=42)

# Create a Surprise dataset
reader = Reader(rating_scale=(1, 5))
data_surprise = Dataset.load_from_df(recommendations_subset[['userId', 'movieId', 'rating']], reader)
trainset = data_surprise.build_full_trainset()  # Build the full trainset

# Function to get the top N recommendations from collaborative filtering results
def get_top_n(predictions, user_id, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n[user_id]

# Function to display recommendations
def display_recommendations(recommendations, subset_df, num_movies=5):
    i = 0
    count = 0
    seen_movies = set()
    while count < num_movies and i < len(recommendations):
        movie_id, est_rating = recommendations[i]

        # Convert the movie ID to an integer
        movie_id = int(movie_id)

        # Check if the movie ID is valid, exists in the subset_df, not already recommended, and not in the skip list
        skip_titles = ['1', '1900']
        matching_rows = subset_df[(subset_df['movieId'] == movie_id) & (~subset_df['movieId'].isin(seen_movies)) & (~subset_df['title'].isin(skip_titles))]
        if not matching_rows.empty:
            movie_info = matching_rows.iloc[0]

            # Display image from hyperlinks column if it's a valid URL
            if pd.notna(movie_info['hyperlinks']) and isinstance(movie_info['hyperlinks'], str):
                try:
                    st.image(movie_info['hyperlinks'], width=200, caption=f"Rating: {est_rating:.2f}")
                except Exception as e:
                    st.warning(f"Failed to display image for Movie ID: {movie_id}. {str(e)}")
            else:
                st.warning(f"No image available for Movie ID: {movie_id}")

            st.write(f"Title: {movie_info['title']}")
            st.write(f"Rating: {est_rating:.2f}")
            st.write(f"Summary (Tagline): {movie_info['tagline']}")
            st.write(f"Overview: {movie_info['overview']}")
            st.write(f"Runtime: {movie_info['runtime']} minutes")
            st.write(f"Genre: {', '.join(eval(movie_info['genres_list']))}")
            count += 1
            seen_movies.add(movie_id)
        else:
            i += 1  # Skip to the next available movie

        if count % 5 == 0:
            st.write("")  # Add an empty line after every 5 movies

    # If there are fewer recommendations than required, randomly select additional movies
    if count < num_movies:
        additional_movies = subset_df[(~subset_df['movieId'].isin(seen_movies)) & (~subset_df['title'].isin(skip_titles))].sample(n=num_movies - count, random_state=42).drop_duplicates('title')
        for _, movie_info in additional_movies.iterrows():
            try:
                st.image(movie_info['hyperlinks'], width=200, caption=f"Rating: Random")
            except Exception as e:
                st.warning(f"Failed to display image for Movie ID: {movie_info['movieId']}. {str(e)}")
            st.write(f"Title: {movie_info['title']}")
            st.write(f"Rating: Random")
            st.write(f"Summary (Tagline): {movie_info['tagline']}")
            st.write(f"Overview: {movie_info['overview']}")
            st.write(f"Runtime: {movie_info['runtime']} minutes")
            st.write(f"Genre: {', '.join(eval(movie_info['genres_list']))}")

# Function for collaborative filtering
def collaborative_search(user_id_collaborative, num_movies=10):
    st.write("Step 1: Preparing to show recommendations...")

    # Collaborative Filtering
    algo = SVD()
    algo.fit(trainset)
    anti_testset = trainset.build_anti_testset()
    user_anti_testset = [(user_id_collaborative, movie_id, 0) for movie_id in anti_testset[user_id_collaborative]]

    # Make collaborative filtering recommendations
    collaborative_recommendations = algo.test(user_anti_testset)
    top_collaborative = get_top_n(collaborative_recommendations, user_id_collaborative, n=num_movies)

    st.write("Step 2: Collaborative Filtering Recommendations...")

    # Display collaborative filtering recommendations
    st.subheader('Top Collaborative Filtering Recommendations:')
    display_recommendations(top_collaborative, recommendations_subset, num_movies)

# Streamlit App for Collaborative Filtering Search
st.title('Collaborative Filtering Search')
# Get user input (userId) for collaborative filtering
user_id_collaborative = st.number_input('Enter your user ID:', min_value=1, max_value=recommendations_subset['userId'].max(), value=1, step=1)
# Button for collaborative filtering
collaborative_button = st.button('Show Collaborative Filtering Recommendations')
if collaborative_button:
    st.write("Collaborative Filtering Process:")
    collaborative_search(user_id_collaborative)
