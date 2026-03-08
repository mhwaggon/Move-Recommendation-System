# Movie Recommender System

A hybrid movie recommendation system built in Python that combines:

- Content-based filtering using TF-IDF on movie genres and user tags
- Collaborative filtering using matrix factorization with Truncated SVD
- A weighted hybrid scoring approach to balance both methods

This project recommends similar movies based on a given movie title.

## Overview

Traditional recommendation systems often rely on either content similarity or user behavior. This project blends both approaches into a single recommender:

- **Content-based filtering** looks at movie metadata like genres and tags
- **Collaborative filtering** looks at user rating behavior across movies
- **Hybrid recommendation** combines both similarity scores into one final ranking

This helps produce recommendations that are both thematically similar and behaviorally relevant.

## Dataset Files

This project expects the following CSV files:

- `ratings.csv`
- `movies.csv`
- `tags.csv`

These files are commonly available in the MovieLens dataset.

### Expected columns

#### `ratings.csv`
- `userId`
- `movieId`
- `rating`

#### `movies.csv`
- `movieId`
- `title`
- `genres`

#### `tags.csv`
- `userId`
- `movieId`
- `tag`

## How It Works

### 1. Load the data
The program reads movie, rating, and tag data from CSV files.

### 2. Build movie metadata
User tags are grouped by `movieId` and merged into the movies dataset.

Then genres and tags are combined into a single text field called `metadata`.

Example:
- Genres: `Adventure|Animation|Children|Comedy|Fantasy`
- Tags: `pixar fun family toys`

Combined metadata:
`Adventure Animation Children Comedy Fantasy pixar fun family toys`

### 3. Convert metadata to numeric features
A `TfidfVectorizer` converts the metadata text into a sparse matrix so movie content can be compared mathematically.

### 4. Build the collaborative filtering matrix
Ratings are pivoted into a user-movie matrix where:
- rows = users
- columns = movies
- values = ratings

Missing ratings are filled with `0`.

### 5. Apply dimensionality reduction
`TruncatedSVD` reduces the user-movie matrix into latent factors that capture hidden patterns in user preferences.

### 6. Generate hybrid recommendations
For a given movie title:
- content similarity is calculated from TF-IDF vectors
- collaborative similarity is calculated from SVD movie factors
- both are blended using a parameter called `alpha`

Formula:

`hybrid_score = alpha * content_score + (1 - alpha) * collaborative_score`

## Recommendation Function

### `recommend_movies(title, top_n=10, alpha=0.5)`

Returns the top recommended movies for a given title.

#### Parameters

- `title` (`str`): Movie title to search for
- `top_n` (`int`): Number of recommendations to return
- `alpha` (`float`): Weight for content-based similarity
  - `1.0` = fully content-based
  - `0.0` = fully collaborative
  - `0.5` = balanced hybrid

#### Returns

A pandas DataFrame with:
- `title`
- `genres`

## Example Usage

```python
user_movie_name = 'Toy Story'
result = recommend_movies(user_movie_name, top_n=10, alpha=0.5)
print(result)
