import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load recommendations.csv
recommendations = pd.read_csv('Dataset/recommendations.csv')

# Sample: Use 1% of the data for faster processing
recommendations_subset = recommendations.sample(frac=0.01, random_state=42)

# Get a set of all genres in the dataset
all_genres = set(genre for genres in recommendations_subset['genres_list'] for genre in genres)

# Streamlit App for Content-Based Filtering Search
st.title('Recommender System - Hybrid Method - Content Boosted Collaborative Filtering')

# Get user input (userId and selected genres) for content-based filtering
user_id_content = st.number_input('Enter your user ID:', min_value=1, max_value=recommendations_subset['userId'].max(), value=1, step=1)

# Selectable genres
selected_genres = st.multiselect('Select genres:', list(all_genres))

# Button for content-based filtering
content_based_button = st.button('Show Content-Based Filtering Recommendations')
if content_based_button:
    st.write("Content-Based Filtering Process:")
    
    # Process the genres for content-based filtering
    recommendations_subset['genres_str'] = recommendations_subset['genres_list'].apply(lambda x: ' '.join(x))

    # Check if the user has rated any movies
    user_rated_movies = recommendations_subset[recommendations_subset['userId'] == user_id_content]

    # Get the index of the user's last movie
    if not user_rated_movies.empty:
        user_last_movie = user_rated_movies.iloc[-1]
        movie_index = recommendations_subset[recommendations_subset['movieId'] == user_last_movie['movieId']].index[0]

        # Create a bag of words from the genres
        count_vectorizer = CountVectorizer()
        genre_matrix = count_vectorizer.fit_transform(recommendations_subset['genres_str'])

        # Compute the cosine similarity matrix
        cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

        # Compute the cosine similarity between the user's last movie and all other movies
        sim_scores = list(enumerate(cosine_sim[movie_index]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Filter movies based on the selected genres
        selected_genre_movies = [index for index, score in sim_scores if any(g.lower() in [genre.lower() for genre in recommendations_subset.iloc[index]['genres_list']] for g in selected_genres)]

        # Display up to 5 movies that match the selected genres
        count = 0
        selected_movies = []
        for movie_index in selected_genre_movies:
            if count >= 5:
                break

            # Check if the movie index is not in the list of selected movies
            if movie_index not in selected_movies:
                selected_movies.append(movie_index)

                movie_info = recommendations_subset.iloc[movie_index]

                # Check if the image URL is valid before displaying
                if not pd.isnull(movie_info['hyperlinks']):
                    st.image(movie_info['hyperlinks'], width=200, caption=f"Rating: {movie_info['vote_average']}")
                else:
                    st.write("Image not available")

                st.write(f"Title: {movie_info['title']}")
                st.write(f"Overview: {movie_info['overview']}")
                st.write(f"Rating: {movie_info['vote_average']}")
                st.write("---")
                count += 1
    else:
        # Initialize a list to keep track of selected movies
        selected_movies = []

        # Get the next set of 5 random movies based on the number of searches made
        start_index = st.session_state.get('search_count', 0) * 5
        end_index = start_index + 5
        random_movies = recommendations_subset[start_index:end_index].sample(frac=1, random_state=42)

        # Iterate over the shuffled DataFrame
        count = 0
        for _, movie_info in random_movies.iterrows():
            # Check if the movie index is not in the list of selected movies
            if movie_info.name not in selected_movies:
                selected_movies.append(movie_info.name)

                # Check if the image URL is valid before displaying
                if not pd.isnull(movie_info['hyperlinks']):
                    st.image(movie_info['hyperlinks'], width=200, caption=f"Rating: {movie_info['vote_average']}")
                else:
                    st.write("Image not available")

                st.write(f"Title: {movie_info['title']}")
                st.write(f"Overview: {movie_info['overview']}")
                st.write(f"Rating: {movie_info['vote_average']}")
                st.write("---")
                count += 1
                if count >= 5:
                    break

        # Increment the search count in the session state
        st.session_state.search_count = st.session_state.get('search_count', 0) + 1
