from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Merged DataFrame as an Excel file
df_movies = pd.read_excel("D:\Dionisio Andres\Desktop\Ciencia de Datos Henry\Movies_Individual_Project_Henry\movies_credits.xlsx")

# Drop rows with null values in specific columns
df_movies.dropna(subset=['collection_name', 'revenue', 'title', 'company_names'], inplace=True)

# Drop duplicate rows based on 'collection_name', 'revenue', and 'title'
df_movies.drop_duplicates(subset=['collection_name', 'revenue', 'title', 'company_names'], inplace=True)

# Convert the 'revenue' column to standard Python int type
df_movies['revenue'] = df_movies['revenue'].astype(int)

# FastAPI app
app = FastAPI()

# Endpoint 1: Movies by Language
@app.get('/movies_by_language/{language}')
def movies_by_language(language: str):
    filtered_movies = df_movies[df_movies['original_language'] == language]
    movie_count = filtered_movies.shape[0]
    return {'language': language, 'count': movie_count}

# Endpoint 2: Movie Duration
@app.get('/movie_duration/{movie_title}')
def movie_duration(movie_title: str):
    movie_info = df_movies[df_movies['title'] == movie_title]
    if movie_info.empty:
        return {'movie': movie_title, 'message': 'Movie not found'}
    duration = movie_info.iloc[0]['runtime']
    year = movie_info.iloc[0]['release_date']
    return {'movie': movie_title, 'duration': duration, 'year': year}

# Endpoint 3: Movie Franchise
@app.get('/franchise/{franchise_name}')
def franchise(franchise_name: str):
    try:
        franchise_info = df_movies[df_movies['collection_name'] == franchise_name]
        if franchise_info.empty:
            return {'franchise': franchise_name, 'message': 'Franchise not found'}, 404  # Not Found status code
        
        movie_count = franchise_info.shape[0]
        total_revenue = int(franchise_info['revenue'].sum())
        average_revenue = int(franchise_info['revenue'].mean())
        
        return {'franchise': franchise_name, 'count': movie_count, 'total_revenue': total_revenue, 'average_revenue': average_revenue}
    except Exception as e:
        return {'error': str(e)}, 500  # Internal Server Error


# Endpoint 4: Movies by Country
@app.get('/movies_by_country/{country}')
def movies_by_country(country: str):
    filtered_movies = df_movies[df_movies['production_countries'].str.contains(country, case=False, na=False)]
    movie_count = filtered_movies.shape[0]
    return {'country': country, 'count': movie_count}

# Endpoint 5: Successful Production Companies
@app.get('/successful_production_companies/{company_name}')
def successful_production_companies(company_name: str):
    company_info = df_movies[df_movies['company_names'].str.contains(company_name, case=False, na=False)]
    if company_info.empty:
        return {'company': company_name, 'message': 'Company not found'}, 404  # Not Found status code
    
    total_revenue = company_info['revenue'].sum()
    movie_count = company_info.shape[0]
    return {'company': company_name, 'total_revenue': total_revenue, 'count': movie_count}

# Endpoint 6: Get Director's Movies
@app.get('/get_director_movies/{director_name}')
def get_director_movies(director_name: str):
    def extract_director_info(crew, job):
        for member in crew:
            if member['job'] == job:
                return member['name']
        return None

    movies_info = []
    for _, row in df_movies.iterrows():
        director_name_found = extract_director_info(row['crew'], 'director')
        if director_name_found and director_name_found.lower() == director_name.lower():
            movie = {
                'title': row['title'],
                'release_date': row['release_date'],
                'individual_return': row['return'],
                'budget': row['budget'],
                'revenue': row['revenue']
            }
            movies_info.append(movie)

    if not movies_info:
        return {'director': director_name, 'message': 'Director not found'}, 404  # Not Found status code

    return movies_info

# Endpoint 7: Movie Recommendation
@app.get('/recommendation/{title}')
def recommendation(title: str):
    try:
        
        title = title.lower()

        # Calculate the TF-IDF matrix for the movie titles
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['title'].str.lower())

        # cosine similarity between the input movie and all other movies
        cosine_similarities = linear_kernel(tfidf_matrix[df_movies[df_movies['title'].str.lower() == title].index], tfidf_matrix).flatten()

        # Indices of the  top 5 similar movies (excluding the input movie itself)
        similar_indices = cosine_similarities.argsort()[:-6:-1]

        # Movie titles of the top 5 similar movies
        recommended_movies = df_movies.iloc[similar_indices]['title'].tolist()

        return recommended_movies
    except Exception as e:
        return {'error': str(e)}, 500  # Internal Server Error



