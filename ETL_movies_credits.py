# %%
import pandas as pd
import numpy as np


# %%
import csv

movies_df = pd.read_excel("D:\Dionisio Andres\Desktop\Ciencia de Datos Henry\Labs - Proyecto - Individual\movies_dataset (1).xlsx")

# %%
# Break down of the dataset 
movies_df.head()

# %%
# Overview of the Columns
print(movies_df.columns)

# %%

# Extracting information from 'belongs_to_collection'
# Define a function to extract 'id', 'name', and 'backdrop_path' from 'belongs_to_collection'
def extract_collection_info(row):
    collection_data = row['belongs_to_collection']
    # Access the 'belongs_to_collection' data for the current row
    if pd.notnull(collection_data):
        # Check if the data is not null
        if isinstance(collection_data, str):
            # Check if the data is a string
            try:
                collection_data = eval(collection_data)
                # Convert the string representation to a dictionary
                collection_id = collection_data.get('id')
                # Get the 'id' value from the dictionary
                collection_name = collection_data.get('name')
                # Get the 'name' value from the dictionary
                backdrop_path = collection_data.get('backdrop_path')
                # Get the 'backdrop_path' value from the dictionary
                return collection_id, collection_name, backdrop_path
            except (SyntaxError, AttributeError):
                # Handle exceptions that might occur during evaluation
                pass
    return None, None, None

# Apply the 'extract_collection_info' function to each row in the DataFrame
movies_df[['collection_id', 'collection_name', 'backdrop_path']] = movies_df.apply(extract_collection_info, axis=1, result_type='expand')

# Display the modified DataFrame with new columns
print(movies_df)




# %%
# Overview of columns
print(movies_df.columns)

# %%

# Define the function to extract 'name' and 'id' values from the list of dictionaries in the 'Production_companies' column
def extract_production_companies_info(row):
    companies_data = row['production_companies']
    # Access the 'production_companies' data for the current row
    if pd.notnull(companies_data):
        # Check if the data is not null
        try:
            companies_data = eval(companies_data)
            # Convert the string representation to a list of dictionaries
            if isinstance(companies_data, list):
                # Check if the data is a list
                company_names = [company['name'] for company in companies_data]
                # Extract 'name' values from each dictionary in the list
                company_ids = [company['id'] for company in companies_data]
                # Extract 'id' values from each dictionary in the list
                return company_names, company_ids
        except (SyntaxError, TypeError):
            # Handle exceptions that might occur during evaluation
            pass
    return None, None

# Apply the 'extract_production_companies_info' function to each row and create new columns 'company_names' and 'company_ids'
movies_df['company_names'], movies_df['company_ids'] = zip(*movies_df.apply(extract_production_companies_info, axis=1))

# Print the modified DataFrame
print(movies_df)




# %%
movies_df.columns

# %%


# Fill NaN values in 'revenue' column with 0
movies_df['revenue'] = movies_df['revenue'].fillna(0)

# Fill NaN values in 'collection_name' column with a placeholder value
movies_df['collection_name'] = movies_df['collection_name'].fillna('Unknown Collection')

# Fill NaN values in 'company_names' column with a placeholder value
movies_df['company_names'] = movies_df['company_names'].fillna('Unknown Company')


# %%

# Exploding values in the 'company_names' and 'company_ids' columns, resulting from the previous 'production_companies' column extraction
movies_df['company_names'] = movies_df['company_names'].explode().reset_index(drop=True)
# Explode the list values in the 'company_names' column and reset the index
movies_df['company_ids'] = movies_df['company_ids'].explode().reset_index(drop=True)
# Explode the list values in the 'company_ids' column and reset the index
print(movies_df)
# Print the modified DataFrame



# %%

# Drop the columns after the extraction and expansion, including:
# 'belongs_to_collection', 'production_companies', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage', 'video'
movies_df.drop(['belongs_to_collection', 'production_companies', 'imdb_id', 'adult', 'original_title', 'poster_path', 'homepage', 'video'], axis=1, inplace=True)
# Drop the specified columns from the DataFrame

# Print the updated DataFrame
print(movies_df)



# %%

# Transformation : Null values in the 'revenue' and 'budget' fields should be filled with the number 0

# Replace null values with 0 in the 'revenue' column
movies_df['revenue'].fillna(0, inplace=True)

# Replace null values with 0 in the 'budget' column
movies_df['budget'].fillna(0, inplace=True)

# Print the new DataFrame after filling null values with 0
print(movies_df)


# %%

# Remove rows with null values in the 'release_date' field
movies_df.dropna(subset=['release_date'], inplace=True)

# Print the DataFrame after removing rows with null values in 'release_date'
print(movies_df)



# %%

# Transformation : If there are dates, they should be in the format YYYY-mm-dd. Additionally, create the 'release_year' column by extracting the year from the release date.

# Convert the 'release_date' column to the pandas date format, handling incorrect values with 'coerce'
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')

# Remove rows with null values in the 'release_date' field
movies_df.dropna(subset=['release_date'], inplace=True)

# Convert the 'release_date' column to the specified date format 'YYYY-mm-dd'
movies_df['release_date'] = movies_df['release_date'].dt.strftime('%Y-%m-%d')

# Print the DataFrame after transformations
print(movies_df)




# %%

# Convert non-numeric values to numeric in the 'budget' and 'revenue' columns
movies_df['budget'] = pd.to_numeric(movies_df['budget'].replace('[^\d]', '', regex=True))
movies_df['revenue'] = pd.to_numeric(movies_df['revenue'].replace('[^\d]', '', regex=True))

# Create the 'return' column with the correct operation, calculating the revenue-to-budget ratio
movies_df['return'] = (movies_df['revenue'] / movies_df['budget']).fillna(0)

# Set the display format for floating-point numbers
pd.options.display.float_format = '{:.2f}'.format

# Print the DataFrame after transformations
print(movies_df)


# %%
import pandas as pd
credits_df = pd.read_csv("D:\Dionisio Andres\Desktop\Ciencia de Datos Henry\Labs - Proyecto - Individual\credits (1).csv")

# %%


# Convert the 'id' column in both DataFrames to numeric type
movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')

# Check for matching 'id' values between movies_df and credits_df
matching_ids = movies_df['id'].isin(credits_df['id'])
# Create a Boolean Series indicating which 'id' values are present in both DataFrames

print("Number of matching 'id' values:", matching_ids.sum())
# Print the count of matching 'id' values


# %%

# Convert the 'id' column in both DataFrames to numeric type
movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce')

# Specify the columns to include in the credits_df
columns_to_include = ['id', 'crew', 'cast']

# Perform the merge
df_movies = pd.merge(movies_df, credits_df[columns_to_include], on='id', how='left')
# Merge movies_df and credits_df using the 'id' column and including specified columns from credits_df



# %%
df_movies.columns

# %%
# Check for NaN values in the 'crew' column
nan_crew = df_movies[df_movies['crew'].isna()]
print("Rows with NaN values in 'crew' column:")
print(nan_crew)

# Check for any other columns with NaN values
nan_columns = df_movies.columns[df_movies.isna().any()].tolist()
print("Columns with NaN values:")
print(nan_columns)


# %%
import pandas as pd

# Replace empty lists in 'crew' column with NaN
df_movies['crew'] = df_movies['crew'].apply(lambda x: pd.NA if x == [] else x)


# %%
# Check if there are any NaN values in the 'crew' column
missing_values_count = df_movies['crew'].isna().sum()

if missing_values_count == 0:
    print("All missing values in 'crew' column have been handled.")
else:
    print("There are still missing values in 'crew' column.")

columns_to_drop = ["cast", "backdrop_path", 'tagline', 'collection_id', 'company_ids', 'status']
df_movies = df_movies.drop(columns_to_drop, axis=1)

# %%
df_movies.to_excel('movies_credits_df.xlsx')

