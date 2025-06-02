#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preprocessing script for Yelp Review Analysis project.
This script cleans and prepares the restaurant review data for analysis and modelling.
"""

import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(file_path):
    """
    Load the Yelp review data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} records.")
    return df

def clean_data(df):
    """
    Clean the data by handling missing values, removing duplicates,
    and converting data types.
    
    Args:
        df (pandas.DataFrame): Raw data
        
    Returns:
        pandas.DataFrame: Cleaned data
    """
    print("Cleaning data...")
    
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Strip whitespace from column names
    df_clean.columns = df_clean.columns.str.strip()
    
    # Convert date column to datetime
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    # Handle missing values
    print(f"Missing values before handling: {df_clean.isnull().sum().sum()}")
    
    # Fill missing review text with empty string
    if 'Review Text' in df_clean.columns:
        df_clean['Review Text'] = df_clean['Review Text'].fillna('')
    
    # Drop rows with missing ratings
    if 'Rating' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['Rating'])
    
    print(f"Missing values after handling: {df_clean.isnull().sum().sum()}")
    
    # Remove duplicates
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate records...")
        df_clean = df_clean.drop_duplicates()
    
    # Sort by date
    df_clean = df_clean.sort_values('Date')
    
    # Extract restaurant name from URL
    if 'Yelp URL' in df_clean.columns:
        df_clean['Restaurant'] = df_clean['Yelp URL'].apply(extract_restaurant_name)
    
    print("Data cleaning completed.")
    return df_clean

def extract_restaurant_name(url):
    """
    Extract restaurant name from Yelp URL.
    
    Args:
        url (str): Yelp URL
        
    Returns:
        str: Restaurant name
    """
    if pd.isna(url):
        return "Unknown"
    
    # Extract restaurant name from URL pattern
    pattern = r'biz/([^/]+)'
    match = re.search(pattern, url)
    if match:
        # Convert hyphenated name to title case
        name = match.group(1).replace('-', ' ').title()
        return name
    else:
        return "Unknown"

def preprocess_text(df):
    """
    Preprocess review text by removing special characters, stopwords,
    and applying lemmatization.
    
    Args:
        df (pandas.DataFrame): Cleaned data with review text
        
    Returns:
        pandas.DataFrame: Data with preprocessed text
    """
    print("Preprocessing review text...")
    
    # Make a copy to avoid modifying the input dataframe
    df_processed = df.copy()
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Function to clean and preprocess text
    def clean_text(text):
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply lemmatization
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        # Join tokens back into text
        cleaned_text = ' '.join(cleaned_tokens)
        
        return cleaned_text
    
    # Apply text preprocessing
    df_processed['Processed_Text'] = df_processed['Review Text'].apply(clean_text)
    
    print("Text preprocessing completed.")
    return df_processed

def engineer_features(df):
    """
    Engineer additional features from the review data.
    
    Args:
        df (pandas.DataFrame): Data with preprocessed text
        
    Returns:
        pandas.DataFrame: Data with engineered features
    """
    print("Engineering features...")
    
    # Make a copy to avoid modifying the input dataframe
    df_featured = df.copy()
    
    # Calculate review length
    df_featured['Review_Length'] = df_featured['Review Text'].apply(lambda x: len(str(x).split()))
    
    # Calculate processed review length
    df_featured['Processed_Length'] = df_featured['Processed_Text'].apply(lambda x: len(str(x).split()))
    
    # Extract year and month from date
    df_featured['Year'] = df_featured['Date'].dt.year
    df_featured['Month'] = df_featured['Date'].dt.month
    
    # Create sentiment category based on rating
    def get_sentiment_category(rating):
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Positive'
    
    df_featured['Sentiment_Category'] = df_featured['Rating'].apply(get_sentiment_category)
    
    # Create binary sentiment (1 for positive, 0 for negative/neutral)
    df_featured['Positive_Sentiment'] = df_featured['Rating'].apply(lambda x: 1 if x >= 4 else 0)
    
    print("Feature engineering completed.")
    return df_featured

def save_processed_data(df, output_path):
    """
    Save the processed data to CSV.
    
    Args:
        df (pandas.DataFrame): Processed data
        output_path (str): Path to save the CSV file
    """
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Data saved successfully.")

def main():
    """Main function to execute the preprocessing pipeline."""
    # Define file paths
    input_path = '../data/YelpRestaurantReviews.csv'
    processed_path = '../data/processed_yelp_reviews.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    # Load data
    df = load_data(input_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Preprocess text
    df_processed = preprocess_text(df_clean)
    
    # Engineer features
    df_featured = engineer_features(df_processed)
    
    # Save processed data
    save_processed_data(df_featured, processed_path)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
