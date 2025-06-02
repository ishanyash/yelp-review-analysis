#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction script for Yelp Review Analysis project.
This script loads the trained model and makes predictions on new reviews.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set yellow color palette for visualisations
YELLOW_PALETTE = ['#FFC300', '#FFD60A', '#FFF176', '#FFEE58', '#F9A825']
sns.set_palette(YELLOW_PALETTE)
plt.style.use('seaborn-v0_8-whitegrid')

def load_model(model_path, vectorizer_path):
    """
    Load the trained model and vectorizer.
    
    Args:
        model_path (str): Path to the model file
        vectorizer_path (str): Path to the vectorizer file
        
    Returns:
        tuple: (model, vectorizer)
    """
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loading vectorizer from {vectorizer_path}...")
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def preprocess_text(text):
    """
    Preprocess review text by removing special characters, stopwords,
    and applying lemmatization.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
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

def predict_sentiment(model, vectorizer, reviews):
    """
    Predict sentiment for a list of reviews.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        reviews (list): List of review texts
        
    Returns:
        pandas.DataFrame: DataFrame with reviews and predictions
    """
    print("Predicting sentiment...")
    
    # Create DataFrame
    df = pd.DataFrame({'Review_Text': reviews})
    
    # Preprocess text
    df['Processed_Text'] = df['Review_Text'].apply(preprocess_text)
    
    # Transform text to features
    X = vectorizer.transform(df['Processed_Text'])
    
    # Make predictions
    df['Predicted_Sentiment'] = model.predict(X)
    
    # Add prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        df['Probability_Positive'] = proba[:, 1]
    
    # Map binary predictions to sentiment categories
    df['Sentiment_Category'] = df['Predicted_Sentiment'].map({0: 'Negative/Neutral', 1: 'Positive'})
    
    print("Predictions completed.")
    return df

def plot_sentiment_distribution(df, output_dir):
    """
    Plot the distribution of predicted sentiments.
    
    Args:
        df (pandas.DataFrame): Data with predictions
        output_dir (str): Directory to save the plot
    """
    print("Plotting sentiment distribution...")
    
    plt.figure(figsize=(10, 6))
    
    # Count sentiment categories
    sentiment_counts = df['Sentiment_Category'].value_counts()
    
    # Plot pie chart
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=YELLOW_PALETTE, startangle=90, explode=[0.05, 0.05])
    
    plt.title('Distribution of Predicted Sentiments', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_sentiment_distribution.png'))
    plt.close()

def plot_probability_distribution(df, output_dir):
    """
    Plot the distribution of prediction probabilities.
    
    Args:
        df (pandas.DataFrame): Data with predictions
        output_dir (str): Directory to save the plot
    """
    if 'Probability_Positive' not in df.columns:
        return
    
    print("Plotting probability distribution...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(df['Probability_Positive'], bins=20, kde=True, color=YELLOW_PALETTE[0])
    
    plt.title('Distribution of Positive Sentiment Probabilities', fontsize=16)
    plt.xlabel('Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()

def save_predictions(df, output_path):
    """
    Save predictions to CSV.
    
    Args:
        df (pandas.DataFrame): Data with predictions
        output_path (str): Path to save the CSV file
    """
    print(f"Saving predictions to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Predictions saved successfully.")

def main():
    """Main function to execute the prediction pipeline."""
    # Define file paths
    model_path = '../models/logistic_regression_model.pkl'  # Adjust based on your best model
    vectorizer_path = '../models/vectorizer.pkl'
    output_dir = '../assets'
    predictions_path = '../data/predictions.csv'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and vectorizer
    model, vectorizer = load_model(model_path, vectorizer_path)
    
    # Example reviews for prediction
    example_reviews = [
        "The food was absolutely delicious and the service was excellent. I will definitely come back!",
        "Average food, nothing special. The prices were a bit high for what you get.",
        "Terrible experience. The food was cold and the staff was rude. Would not recommend.",
        "Great atmosphere and decent food. The cocktails were amazing!",
        "The restaurant was clean and the staff was friendly, but the food was just okay."
    ]
    
    # Make predictions
    df_pred = predict_sentiment(model, vectorizer, example_reviews)
    
    # Plot sentiment distribution
    plot_sentiment_distribution(df_pred, output_dir)
    
    # Plot probability distribution
    plot_probability_distribution(df_pred, output_dir)
    
    # Save predictions
    save_predictions(df_pred, predictions_path)
    
    print("Prediction process completed successfully!")

if __name__ == "__main__":
    main()
