import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from the csv file to a pandas dataframe
movies_data = pd.read_csv("movies.csv")

# printing the first 5 rows of the dataframe
movies_data.head()

# number of rows and columns in the data frame
movies_data.shape

# selecting the relevant features for recommendation
selected_features = ["genres", "keywords", "tagline", "cast", "director"]

# replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna("")

# combining all the 5 selected features
combined_features = (
    movies_data["genres"]
    + " "
    + movies_data["keywords"]
    + " "
    + movies_data["tagline"]
    + " "
    + movies_data["cast"]
    + " "
    + movies_data["director"]
)


# converting the text data to feature vectors
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

# getting the movie name from the user
movie_name = input("Enter your favorite game name : ")

# creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data["title"].tolist()

# finding the close match for the movie name given by the user
list_of_close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)
# print(list_of_close_matches)

best_match = list_of_close_matches[0]
# print(close_match)

# finding the index of the movie with title
index_of_the_movie = movies_data[movies_data.title == best_match]["index"].values[0]
# print(index_of_the_movie)

# getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))
# print(similarity_score)

# len(similarity_score)

# sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
# print(sorted_similar_movies)

# print the name of similar movies based on the index
print("Games suggested for you : \n")
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    movie_title = movies_data[movies_data.index == index]["title"].values[0]
    production_companies = eval(
        movies_data[movies_data.index == index]["production_companies"].values[0]
    )
    if i > 30:
        break
    print(f"Game {i}: {movie_title}")
    print(f"Production Companies:")
    cnt = 0
    for company in production_companies:
        print(f"   {cnt+1}. {company['name']}")
        cnt += 1
    print()
    i += 1
