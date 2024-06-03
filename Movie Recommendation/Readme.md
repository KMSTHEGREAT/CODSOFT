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
```

## 2. Data Preparation

Next, we prepare the data by creating a user-item interaction matrix and filtering it based on the number of ratings.

```python

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
```
## 3. Model Training

We then create and train a K-Nearest Neighbors (KNN) model using the prepared data.

```python

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Create and train the KNN model
KNN = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=20, n_jobs=-1)
KNN.fit(csr_data)
```
## 4. Recommendation Function

Finally, we define a function to get movie recommendations based on the trained KNN model.

```python

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
```
# Test the recommendation system
print(get_movie_recommendation('Iron Man'))

# Output
  <div id="df-63d4d875-fe50-40c2-994e-9c6fb26df092" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Up (2009)</td>
      <td>0.368857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Guardians of the Galaxy (2014)</td>
      <td>0.368758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Watchmen (2009)</td>
      <td>0.368558</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Star Trek (2009)</td>
      <td>0.366029</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Batman Begins (2005)</td>
      <td>0.362759</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Avatar (2009)</td>
      <td>0.310893</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Iron Man 2 (2010)</td>
      <td>0.307492</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WALL·E (2008)</td>
      <td>0.298138</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dark Knight, The (2008)</td>
      <td>0.285835</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Avengers, The (2012)</td>
      <td>0.285319</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-63d4d875-fe50-40c2-994e-9c6fb26df092')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-63d4d875-fe50-40c2-994e-9c6fb26df092 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-63d4d875-fe50-40c2-994e-9c6fb26df092');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-c5cd3599-4f0d-48d7-a785-7249f3650669">
  <button class="colab-df-quickchart" onclick="quickchart('df-c5cd3599-4f0d-48d7-a785-7249f3650669')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-c5cd3599-4f0d-48d7-a785-7249f3650669 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>

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

```bash

pip install pandas numpy matplotlib seaborn scikit-learn scipy
```
Dataset

The MovieLens dataset used in this project can be downloaded from the following links:


    [!ml-latest-small][https://files.grouplens.org/datasets/movielens/ml-latest-small.zip]


Feel free to modify the `README.md` file according to your specific needs or environment.
