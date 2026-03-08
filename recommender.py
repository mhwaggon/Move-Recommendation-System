# get data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")


# get all movie tags
tag_agg = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

#join dataset to tags
movies = movies.merge(tag_agg, on='movieId', how='left')

# fill blanks
movies['tag'] = movies['tag'].fillna('')
movies['genres'] = movies['genres'].fillna('')

# clean | and create a col for all the metadata for each movie
movies['metadata'] = movies['genres'].str.replace('|', ' ', regex=False) + ' ' + movies['tag']


# using tfidf_vecotr to make numerical features for metadata
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['metadata'])


# unique index and title lookup. This is just faster
title_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()


# collab filtering 
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# dimensionality reduction with SVD
svd = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = svd.fit_transform(user_movie_matrix)
movie_factors = svd.components_.T 

# store movie ids with ratings in collab matrix
cf_movie_ids = user_movie_matrix.columns.tolist()

# find corresponding row in the movies dataframe
cf_movie_indices = [movies[movies['movieId'] == mid].index[0] for mid in cf_movie_ids]


def recommend_movies(title, top_n=10, alpha=0.5):
    
    # check for movie to make sure it exists
    if title not in title_index:
        print("Movie not found")
        return pd.DataFrame()
        
    # get movie and indx
    content_idx = title_index[title]
    movie_id = movies.iloc[content_idx]['movieId']
    if movie_id not in cf_movie_ids:
        print("Movie not found in collab filtering matrix.")
        return pd.DataFrame()

    # compare tfidf vector with other movies
    content_vec = tfidf_matrix[content_idx]
    content_scores = cosine_similarity(content_vec, tfidf_matrix[cf_movie_indices]).flatten()

    # compare svd vector to others
    cf_idx = cf_movie_ids.index(movie_id)
    cf_vec = movie_factors[cf_idx].reshape(1, -1)
    cf_scores = cosine_similarity(cf_vec, movie_factors).flatten()

    # combine both similarities. alpha = content, 1 - alpha = collab. Both weighted to .5
    hybrid_scores = alpha * content_scores + (1 - alpha) * cf_scores

    # get top and dont include input movie
    recommended_indices = np.argsort(hybrid_scores)[::-1]
    recommended_indices = [i for i in recommended_indices if cf_movie_indices[i] != content_idx][:top_n]

    
    return movies.iloc[[cf_movie_indices[i] for i in recommended_indices]][['title', 'genres']].reset_index(drop=True)



# Example usage
user_movie_name = 'Toy Story'
result = recommend_movies(f"{user_movie_name}", top_n=10, alpha=0.5)
print(result)



