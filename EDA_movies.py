# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
 

# %% [markdown]
# 

# %%

df_movies = pd.read_excel("D:\Dionisio Andres\Desktop\Ciencia de Datos Henry\Movies_Individual_Project_Henry\movies_credits.xlsx")

# Display the first few rows of the 'df_movies' DataFrame.
print(df_movies.head())

# Display general statistics for the numeric columns in the DataFrame.
print(df_movies.describe())

# Check the data types and presence of null values in the DataFrame.
print(df_movies.info())



# %%
df_movies.shape

# %%
# Check for duplicates
duplicates = df_movies.duplicated()
print(f"Number of Duplicated Rows: {duplicates.sum()}")

# Check for null values
null_values = df_movies.isnull().sum()
print(f"Number of Null Values: {null_values}")

# Fill null values
for column in df_movies.columns:
    if df_movies[column].isnull().sum() > 0:
        df_movies[column].fillna(value="", inplace=True)

# Print the cleaned data
print(df_movies.head())


# %%
# Remove duplicate Rows
df_movies.drop_duplicates(inplace=True)


# %%

# Convert numeric columns to numeric data type
numeric_columns = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
df_movies[numeric_columns] = df_movies[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Check for duplicates
duplicates = df_movies.duplicated()
print(f"Number of Duplicated Rows: {duplicates.sum()}")

# Check for null values
null_values = df_movies.isnull().sum()
print(f"Number of Null Values: {null_values}")

# Fill null values with the mean
for column in numeric_columns:
    if df_movies[column].isnull().sum() > 0:
        df_movies[column].fillna(df_movies[column].mean(), inplace=True)

print(df_movies.head())



# %%
# For categorical columns, fill null values with the most frequent category.
categorical_columns = ['original_language', 'production_countries']
for column in categorical_columns:
    most_frequent_category = df_movies[column].mode()[0]
    df_movies[column].fillna(most_frequent_category, inplace=True)

# %%
# Convert 'release_date' column to datetime type
df_movies['release_date'] = pd.to_datetime(df_movies['release_date'])


# %%
# Calculate the average budget for movies
average_budget = df_movies['budget'].mean()
print(f"The average budget per movie is ${average_budget:.2f}")


# %%

# Create a boxplot to analyze outliers in the 'budget' column
plt.figure(figsize=(10, 6))
sns.boxplot(df_movies['budget'], palette='viridis')
plt.title('Budget Boxplot')
plt.xlabel('Budget')
plt.show()


# %%

# Calculate standard deviations for budget and revenue
budget_std = df_movies['budget'].std()
revenue_std = df_movies['revenue'].std()

# Set up the figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot histogram for budget
sns.histplot(df_movies['budget'], bins=30, ax=axes[0], color='blue')
axes[0].axvline(df_movies['budget'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
axes[0].axvline(df_movies['budget'].mean() + budget_std, color='orange', linestyle='dashed', linewidth=2, label='Mean + Std Dev')
axes[0].axvline(df_movies['budget'].mean() - budget_std, color='orange', linestyle='dashed', linewidth=2, label='Mean - Std Dev')
axes[0].set_title('Budget Distribution')
axes[0].legend()

# Plot histogram for revenue
sns.histplot(df_movies['revenue'], bins=30, ax=axes[1], color='green')
axes[1].axvline(df_movies['revenue'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
axes[1].axvline(df_movies['revenue'].mean() + revenue_std, color='orange', linestyle='dashed', linewidth=2, label='Mean + Std Dev')
axes[1].axvline(df_movies['revenue'].mean() - revenue_std, color='orange', linestyle='dashed', linewidth=2, label='Mean - Std Dev')
axes[1].set_title('Revenue Distribution')
axes[1].legend()

# Adjust layout and display
plt.tight_layout()
plt.show()




# %%
#Distrtibutio of Sample Means for Revenue
# Initialize variables
sample_means = []
sample_size = 50
num_samples = 1000

# Generate sample means for 1000 samples of size 50
for _ in range(num_samples):
    sample = np.random.choice(df_movies['revenue'], size=sample_size, replace=False)
    sample_means.append(np.mean(sample))

# Plot the distribution of sample means
plt.figure(figsize=(8, 6))
sns.histplot(sample_means, kde=True)
plt.title(f'Distribution of Sample Means (Sample Size = {sample_size})')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.show()



# %%
# Calculate correlation between budget and revenue
correlation_budget_revenue = df_movies['budget'].corr(df_movies['revenue'])
print(f"Correlation between Budget and Revenue: {correlation_budget_revenue:.2f}")


# %%

# Create a joint plot to visualize the relationship between budget and revenue
sns.jointplot(data=df_movies, x='budget', y='revenue', kind='scatter', height=8)
plt.show()

# The plot shows a positive relationship between revenue and budget; higher budget could indicate higher revenue.


# %%

# Define a function to extract genre names from genres_list
def extract_genre_name(genres_list):
    genre_names = [genre['name'] for genre in genres_list]
    return genre_names

# Apply the function to 'genres' column and create a new 'genre_names' column
df_movies['genre_names'] = df_movies['genres'].apply(lambda x: extract_genre_name(eval(x)))


# %%

# Count occurrences of each genre
genres_counts = df_movies['genre_names'].explode().value_counts()

# Plot bar chart for the most popular genres
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
sns.barplot(x=genres_counts.index, y=genres_counts.values, palette='viridis')
plt.title('Genero de Peliculas Mas Populares')
plt.xlabel('Genero')
plt.ylabel('Numero')
plt.xticks(rotation=45, ha='right')

plt.savefig('Genero Mas Populares.png', dpi=300, bbox_inches='tight')

plt.show()


# %%

# Relationship between Budget and revenue

# Explode the 'genre_names' column to individual rows
movies_df_exploded = df_movies.explode('genre_names')


top_10_genres = movies_df_exploded['genre_names'].value_counts().nlargest(10).index


movies_df_filtered = movies_df_exploded[movies_df_exploded['genre_names'].isin(top_10_genres)]


movies_df_filtered['revenue'] = movies_df_filtered['revenue'] / 1e9
movies_df_filtered['budget'] = movies_df_filtered['budget'] / 1e8

# Plot the relationship between budget and revenue for movies of the top 10 genres
plt.figure(figsize=(12, 6))
sns.scatterplot(data=movies_df_filtered, x='budget', y='revenue', hue='genre_names', palette='coolwarm')
plt.title('Relationship between Budget and Revenue for Movies of the Top 10 Genres')
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()


# %%

# Scatterplot between 'Popularity' and 'Revenue'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_movies, x='popularity', y='revenue', alpha=0.7, color='blue')
plt.title('Relationship between Popularity and Revenue')
plt.xlabel('Popularity')
plt.ylabel('Revenue')
plt.show()


# %%

# Box plot for the relationship between Genre and Revenue
plt.figure(figsize=(12, 6))
movies_df_exploded = df_movies.explode('genre_names')
sns.boxplot(data=movies_df_exploded, x='genre_names', y='revenue', palette='tab20')
plt.title('Boxplot of Revenue by Genre')
plt.xlabel('Genre')
plt.ylabel('Revenue')
plt.xticks(rotation=45, ha='right')
plt.show()

# Top 10 genres
top_10_genres = movies_df_exploded['genre_names'].value_counts().sort_values(ascending=False).index[:10]

# Violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=movies_df_exploded[movies_df_exploded['genre_names'].isin(top_10_genres)], x='genre_names', y='revenue', palette='Set3', width=1.5)

plt.title('Violin Plot of Revenue by Genre', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Revenue', fontsize=14)

plt.xticks(rotation=45, ha='right')

plt.show()


# %%

# Time Series Analysis of Revenue Over Time
plt.figure(figsize=(12, 6))


df_movies['release_date'] = pd.to_datetime(df_movies['release_date'])
df_movies.set_index('release_date')['revenue'].plot()


plt.title('Revenue Over Time')
plt.xlabel('Release Date')
plt.ylabel('Revenue')

plt.show()


# %%

# Revenue by Original Language
df_movies['release_date'] = pd.to_datetime(df_movies['release_date'])


def convert_to_numeric(x):
    try:
        return float(x)
    except:
        return None


df_movies['revenue'] = df_movies['revenue'].apply(convert_to_numeric)

# Function to clean 'original_language'
def clean_original_language(x):
    if isinstance(x, str):
        return x.strip().lower()  
    return None

# Cleaning Data
df_movies['original_language'] = df_movies['original_language'].apply(clean_original_language)

# Drop  missing values for revenue
df_movies.dropna(subset=['revenue'], inplace=True)

# Group by original_language and calculate summary statistics
revenue_summary = df_movies.groupby('original_language')['revenue'].agg(['mean', 'median', 'max', 'min'])

#  Top 5 languages based on mean revenue
top_5_languages = revenue_summary.nlargest(5, 'mean')

# Bar plot with error bars
plt.figure(figsize=(12, 6))
sns.barplot(x=top_5_languages.index, y='mean', data=top_5_languages, palette='coolwarm')
plt.errorbar(x=top_5_languages.index, y=top_5_languages['mean'], yerr=top_5_languages['max'] - top_5_languages['min'], 
             fmt='none', color='black', capsize=5)
plt.title('Revenue by Original Language (Top 5)')
plt.xlabel('Original Language')
plt.ylabel('Revenue')
plt.xticks(rotation=45, ha='right')
plt.show()


# %%

# Common Release Dates for Movies of Different Genres:

# Convert 'release_date' to datetime format
df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], format='%Y-%b-%d')

# Extract Year and Month
df_movies['release_month'] = df_movies['release_date'].dt.month
df_movies['release_year'] = df_movies['release_date'].dt.year

# Explode the 'genre_names' column to individual rows
movies_df_exploded = df_movies.explode('genre_names')

# Group by genre and release month to get the most common release months
common_release_months = movies_df_exploded.groupby('genre_names')['release_month'].agg(lambda x: x.mode().values[0])


plt.figure(figsize=(12, 6))
sns.barplot(x=common_release_months.index, y=common_release_months.values, palette='plasma')
plt.title('Most Common Release Months by Genre')
plt.xlabel('Genre')
plt.ylabel('Most Common Release Month')
plt.xticks(rotation=45, ha='right')
plt.show()


# %%

# Heatmap to represent the percentage of movies released per month according to genre.


df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], format='%Y-%b-%d')

# Extract Year and Month
df_movies['release_month'] = df_movies['release_date'].dt.month
df_movies['release_year'] = df_movies['release_date'].dt.year

# Explode the 'genre_names' column to individual rows
movies_df_unnested = df_movies.explode('genre_names')

# Count movies per genre and  per month
movies_per_genre_month = movies_df_unnested.groupby(['genre_names', 'release_month'])['id'].count().reset_index()
movies_per_genre_month.rename(columns={'id': 'movie_count'}, inplace=True)

# Count total movies per genre
total_movies_per_genre = movies_df_unnested.groupby('genre_names')['id'].count().reset_index()
total_movies_per_genre.rename(columns={'id': 'total_movie_count'}, inplace=True)

# Merge the two dataframes and calculate the percentage
movies_per_genre_month = movies_per_genre_month.merge(total_movies_per_genre, on='genre_names', how='left')
movies_per_genre_month['percentage'] = (movies_per_genre_month['movie_count'] / movies_per_genre_month['total_movie_count']) * 100

# Create a pivot table for heatmap data
heatmap_data = movies_per_genre_month.pivot(index='genre_names', columns='release_month', values='percentage')

# Create a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.1f', cbar_kws={'label': 'Percentage'})
plt.title('Percentage of Movies Released by Genre Each Month')
plt.xlabel('Month')
plt.ylabel('Genre')
plt.xticks(rotation=45, ha='right')
plt.show()


# %%
# Average movie runtime by genre

df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], format='%Y-%b-%d')

# Extract Year and Month
df_movies['release_month'] = df_movies['release_date'].dt.month
df_movies['release_year'] = df_movies['release_date'].dt.year

# Ensure 'runtime' column contains only numeric values
df_movies['runtime'] = pd.to_numeric(df_movies['runtime'], errors='coerce')  # Convert non-numeric values to NaN

# Explode the 'genre_names' column to individual rows
movies_df_exploded = df_movies.explode('genre_names')

# Calculate average runtime by genre
average_runtime_by_genre = movies_df_exploded.groupby('genre_names')['runtime'].mean()

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=average_runtime_by_genre.index, y=average_runtime_by_genre.values, palette='magma')
plt.title('Average Movie Runtime by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Runtime (minutes)')
plt.xticks(rotation=45, ha='right')
plt.show()



