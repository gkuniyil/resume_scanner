# Resume Matcher API

This project is a lightweight machine learning API built with FastAPI to rank resumes based on how well they match a given job description. It uses TF-IDF vectorization and cosine similarity to determine relevance. Designed for modularity and extensibility.

## Features

- Resume ranking based on semantic similarity to a job description
- FastAPI backend with OpenAPI documentation
- Preprocessed vector storage using Pickle files
- Lightweight testing script for command-line validation

## How It Works

1. Resume content is vectorized using scikit-learn's TfidfVectorizer.
2. A job description is similarly vectorized.
3. Cosine similarity is computed between the job description and each resume vector.
4. The top-k ranked resumes are returned.

## Project Structure

