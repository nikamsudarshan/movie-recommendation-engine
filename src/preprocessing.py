import re
import pandas as pd
import numpy as np

def clean_text(text):
    """
    Applies lowercasing, strips punctuation, and normalizes whitespaces.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase conversion
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Strip unnecessary whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def build_metadata_soup(df):
    """
    Combines core descriptive text attributes into a single feature column.
    Assumes incoming dataframe has 'overview' and 'genres' columns.
    """
    df = df.copy()

    # Fill structural missing values with empty strings
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('')

    # Combine individual features into a single metadata profile
    df['metadata_soup'] = df['overview'] + " " + df['genres']

    # Apply text cleaning pipeline
    df['cleaned_soup'] = df['metadata_soup'].apply(clean_text)

    return df

def generate_interaction_matrix(ratings_df):
    """
    Converts a long ratings dataframe into a sparse pivot table for KNN.
    Rows: User IDs, Columns: Movie IDs/Titles, Values: Ratings.
    """
    # Create the matrix using a pivot table operation
    interaction_matrix = ratings_df.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    )

    # Fill unrated items with 0 to complete the spatial grid for distance metrics
    interaction_matrix = interaction_matrix.fillna(0)

    return interaction_matrix
