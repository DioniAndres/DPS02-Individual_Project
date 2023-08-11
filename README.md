
# **Movie Recommendation Project**
This repository contains a project that involves extracting, transforming, and loading (ETL) data from two datasets, performing exploratory data analysis (EDA) on the merged data, creating an API with meaningful insights using FastAPI, and implementing a movie recommendation system using machine learning techniques.

## **Project Overview**
The main goal of this project is to develop a movie recommendation system based on machine learning. The project involves several key steps, including data extraction, transformation, and loading (ETL) of two datasets with nested values. The cleaned and transformed data is merged, and insights are extracted through an API using FastAPI. Additionally, an exploratory data analysis (EDA) is conducted to gain insights into the data distribution and relationships. Finally, a machine learning model is implemented to create a movie recommendation system.
## **ETL Process**
1. Extract: Data is extracted from two datasets containing movie-related information. These datasets have nested values that require special handling during extraction.
1. Transform: The extracted data is cleaned and transformed to handle missing values, convert data types, and process nested values into a suitable format for analysis and modeling.
1. Load: The cleaned and transformed data from both datasets is merged into a single dataset, providing a comprehensive view of movie-related information.
## **API Development**
- Seven different functions are created, each representing a specific analysis or insight that can be obtained from the merged dataset.
- The FastAPI framework is used to create an API, and each function is decorated with the @app.get endpoint to enable easy access to insights.
## **Exploratory Data Analysis (EDA)**
- The EDA process involves analyzing various aspects of the dataset, including genres, popularity, budget, revenue, and more.
- Insights are visualized using graphs and charts to facilitate understanding of data distribution and relationships.
## **Movie Recommendation System**
- A machine learning model is implemented to create a movie recommendation system.
- The model calculates movie similarity based on features such as movie titles, overviews, and vote averages using TF-IDF vectorization and cosine similarity.
- The recommendation system takes a movie title as input and provides a list of top 5 recommended movies.
## **Conclusion**
This project showcases the entire data pipeline, from ETL processes to API development, EDA, and machine learning implementation. By leveraging data analysis and machine learning techniques, meaningful insights are extracted from the data, and a movie recommendation system is built to enhance user experience and engagement with movie content.

For detailed implementation and code examples, refer to the provided notebooks and source code files in this repository.
