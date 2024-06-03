# Collaborative Filtering Movie Recommendation System

This project implements a movie recommendation system using collaborative filtering with K-Nearest Neighbors (KNN).

## 1. Loading Data

First, we need to load the necessary datasets.

```python
# Import libraries
import pandas as pd
import numpy as np

# Downloading MovieLens dataset

!wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip

# Load datasets
movies = pd.read_csv("/content/ml-latest-small/movies.csv")
ratings = pd.read_csv("/content/ml-latest-small/ratings.csv")

# Display first few rows
print(ratings.head())
print(movies.head())
```python

## 2. Data Preparation

Next, we prepare the data by creating a user-item interaction matrix and filtering it based on the number of ratings.

python

# Create a user-item interaction matrix
combined_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
combined_dataset.fillna(0, inplace=True)

# Filter movies and users based on the number of ratings
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# Visualize the distribution
import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(1, 1, figsize=(16, 4))
plt.scatter(no_user_voted.index, no_user_voted, color='mediumseagreen')
plt.axhline(y=10, color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()

combined_dataset = combined_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
combined_dataset = combined_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

csr_data = csr_matrix(combined_dataset.values)
combined_dataset.reset_index(inplace=True)

## 3. Model Training

We then create and train a K-Nearest Neighbors (KNN) model using the prepared data.

python

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Create and train the KNN model
KNN = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=20, n_jobs=-1)
KNN.fit(csr_data)

## 4. Recommendation Function

Finally, we define a function to get movie recommendations based on the trained KNN model.

python

def get_movie_recommendation(movie_name):
    n_movies_to_recommend = 10
    movie_list = movies[movies['title'].str.contains(movie_name, case=False)]
    
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = combined_dataset[combined_dataset['movieId'] == movie_idx].index[0]

        distances, indices = KNN.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend+1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1:]
        
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = combined_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend+1))
        return df
    else:
        return "No movies found. Please check your input"

# Test the recommendation system
print(get_movie_recommendation('Iron Man'))

Running the System

To run the system, ensure that all the above code blocks are executed in sequence in your Python environment.
Dependencies

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    scipy

Ensure that these packages are installed in your Python environment. You can install them using pip:

bash

pip install pandas numpy matplotlib seaborn scikit-learn scipy

Dataset

The MovieLens dataset used in this project can be downloaded from the following links:


    ml-latest-small

css


Feel free to modify the `README.md` file according to your specific needs or environment.
