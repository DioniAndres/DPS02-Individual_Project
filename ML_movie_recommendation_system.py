# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack  

# %%
df_movies = pd.read_excel("D:\Dionisio Andres\Desktop\Ciencia de Datos Henry\Movies_Individual_Project_Henry\movies_credits.xlsx")


# %%
df_movies.columns

# %%
df_movies = df_movies.drop(columns=['Unnamed: 0'])

# %%
df_movies.info

# %%
df_movies['vote_average'].describe

# %%
# Convert the 'vote_average' column to numeric type, handling any errors by coercing non-numeric values to NaN
df_movies['vote_average'] = pd.to_numeric(df_movies['vote_average'], errors='coerce')

# %%

# Drop rows from df_movies where the 'vote_average' column contains missing values (NaN)
df_movies = df_movies.dropna(subset=['vote_average'])

# %%
# Extract the 'vote_average' values from df_movies and reshape them into a 2D array with a single column
vote_average_values = df_movies['vote_average'].values.reshape(-1, 1).astype(float)

# %%
df_movies['title'].describe

# %%
# Check for invalid elements in the 'title' column
for title in df_movies['title']:
    if not isinstance(title, str):
        print(f"Invalid value in 'title' column: {title}")

# %%
# Filter the DataFrame to exclude rows with invalid values in the 'title' column
df_movies = df_movies[df_movies['title'].apply(lambda x: isinstance(x, str))]

# %% [markdown]
# Movies Recommendation System machine learning Application

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack  

# Basic overview of the data
print(df_movies.info())
print(df_movies.describe())

# Word cloud of movie titles
title_text = ' '.join(df_movies['title'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(title_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Movie Titles')
plt.axis('off')
plt.show()

# Distribution of movie durations
plt.figure(figsize=(10, 6))
sns.histplot(df_movies['runtime'], bins=30, kde=True)
plt.title('Distribution of Movie Durations')
plt.xlabel('Duration (minutes)')
plt.ylabel('Count')
plt.show()

# TF-IDF Vectorization for movie titles and overviews
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_titles = tfidf_vectorizer.fit_transform(df_movies['title'].str.lower())
tfidf_matrix_overviews = tfidf_vectorizer.fit_transform(df_movies['overview'].fillna('').str.lower())

# Combine TF-IDF matrices with movie ratings for recommendations
tfidf_matrix_combined = hstack([tfidf_matrix_titles, tfidf_matrix_overviews])
vote_average_values = df_movies['vote_average'].values.reshape(-1, 1).astype(float)  # Convert 'vote_average' to float type
tfidf_matrix_combined = tfidf_matrix_combined.multiply(vote_average_values)

# It's not necessary to calculate cosine similarity for all movies
# Instead, we will calculate similarity only for the input movie and get the top 5 similar movies
def movie_recommendation(title):
    # Ensure the movie title is in lowercase for case-insensitive matching
    title = title.lower()

    # Get the index of the movie with the provided title
    movie_index = df_movies[df_movies['title'].str.lower() == title].index[0]

    # Calculate cosine similarity between the input movie and all other movies
    similarity_scores = cosine_similarity(tfidf_matrix_combined.getrow(movie_index), tfidf_matrix_combined)

    # Get indices of the top 5 similar movies (excluding the input movie)
    similar_movie_indices = similarity_scores.argsort()[0][-6:-1][::-1]

    # Get titles of the top 5 similar movies
    recommended_movies = df_movies.iloc[similar_movie_indices]['title'].tolist()

    return recommended_movies

# Movie Recommendations
input_movie = 'Toy Story'
recommended_movies = movie_recommendation(input_movie)

print(f"Recommended Movies for '{input_movie}':")
for idx, movie in enumerate(recommended_movies, 1):
    print(f"{idx}. {movie}")


