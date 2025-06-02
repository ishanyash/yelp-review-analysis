#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training script for Yelp Review Analysis project.
This script trains classification models for review sentiment analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Set yellow color palette for visualisations
YELLOW_PALETTE = ['#FFC300', '#FFD60A', '#FFF176', '#FFEE58', '#F9A825']
sns.set_palette(YELLOW_PALETTE)
plt.style.use('seaborn-v0_8-whitegrid')

def load_data(file_path):
    """
    Load the processed review data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Successfully loaded {len(df)} records.")
    return df

def split_data(df, test_size=0.2):
    """
    Split the data into training and testing sets.
    
    Args:
        df (pandas.DataFrame): Data with features
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Splitting data into training and testing sets...")
    
    # Determine the split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split the data
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    # Define features and target
    X_train = train_data['Processed_Text']
    y_train = train_data['Positive_Sentiment']
    X_test = test_data['Processed_Text']
    y_test = test_data['Positive_Sentiment']
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def extract_features(X_train, X_test):
    """
    Extract features from text using TF-IDF vectorisation.
    
    Args:
        X_train (pandas.Series): Training text data
        X_test (pandas.Series): Testing text data
        
    Returns:
        tuple: (X_train_features, X_test_features, vectorizer)
    """
    print("Extracting features using TF-IDF...")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7)
    
    # Fit on training data and transform both training and testing data
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    print(f"Number of features: {X_train_features.shape[1]}")
    
    return X_train_features, X_test_features, vectorizer

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train (scipy.sparse.csr.csr_matrix): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained model
    """
    print("Training Random Forest classifier...")
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                              cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Training accuracy: {grid_search.best_score_:.4f}")
    
    return best_rf

def train_naive_bayes(X_train, y_train):
    """
    Train a Multinomial Naive Bayes classifier.
    
    Args:
        X_train (scipy.sparse.csr.csr_matrix): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        sklearn.naive_bayes.MultinomialNB: Trained model
    """
    print("Training Naive Bayes classifier...")
    
    # Define parameter grid for grid search
    param_grid = {
        'alpha': [0.1, 0.5, 1.0]
    }
    
    # Initialize Naive Bayes classifier
    nb = MultinomialNB()
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, 
                              cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_nb = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Training accuracy: {grid_search.best_score_:.4f}")
    
    return best_nb

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    
    Args:
        X_train (scipy.sparse.csr.csr_matrix): Training features
        y_train (pandas.Series): Training target
        
    Returns:
        sklearn.linear_model.LogisticRegression: Trained model
    """
    print("Training Logistic Regression classifier...")
    
    # Define parameter grid for grid search
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2']
    }
    
    # Initialize Logistic Regression classifier
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, 
                              cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_lr = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Training accuracy: {grid_search.best_score_:.4f}")
    
    return best_lr

def evaluate_model(model, X_test, y_test, model_name, output_dir):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model: Trained model
        X_test (scipy.sparse.csr.csr_matrix): Testing features
        y_test (pandas.Series): Testing target
        model_name (str): Name of the model
        output_dir (str): Directory to save evaluation results
        
    Returns:
        float: Test accuracy
    """
    print(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    print(f"Classification report:\n{report}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', 
                xticklabels=['Negative/Neutral', 'Positive'], 
                yticklabels=['Negative/Neutral', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.close()
    
    # Save classification report to file
    with open(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_report.txt'), 'w') as f:
        f.write(f"# {model_name} Evaluation\n\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
    
    return accuracy

def get_top_features(vectorizer, model, n=20):
    """
    Get top features for Logistic Regression model.
    
    Args:
        vectorizer: TF-IDF vectorizer
        model: Trained model
        n (int): Number of top features to return
        
    Returns:
        tuple: (positive_features, negative_features)
    """
    # Check if model has coef_ attribute (Logistic Regression)
    if hasattr(model, 'coef_'):
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get coefficients
        coef = model.coef_[0]
        
        # Get top positive and negative features
        top_positive_idx = coef.argsort()[-n:][::-1]
        top_negative_idx = coef.argsort()[:n]
        
        top_positive = [(feature_names[i], coef[i]) for i in top_positive_idx]
        top_negative = [(feature_names[i], coef[i]) for i in top_negative_idx]
        
        return top_positive, top_negative
    else:
        return None, None

def plot_top_features(top_positive, top_negative, model_name, output_dir):
    """
    Plot top positive and negative features.
    
    Args:
        top_positive (list): List of (feature, coefficient) tuples for positive features
        top_negative (list): List of (feature, coefficient) tuples for negative features
        model_name (str): Name of the model
        output_dir (str): Directory to save the plot
    """
    if top_positive is None or top_negative is None:
        return
    
    print(f"Plotting top features for {model_name}...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot top positive features
    features, coefficients = zip(*top_positive)
    y_pos = np.arange(len(features))
    ax1.barh(y_pos, coefficients, color=YELLOW_PALETTE[0])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.invert_yaxis()
    ax1.set_title('Top Positive Features', fontsize=16)
    ax1.set_xlabel('Coefficient', fontsize=12)
    
    # Plot top negative features
    features, coefficients = zip(*top_negative)
    y_pos = np.arange(len(features))
    ax2.barh(y_pos, coefficients, color=YELLOW_PALETTE[1])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features)
    ax2.invert_yaxis()
    ax2.set_title('Top Negative Features', fontsize=16)
    ax2.set_xlabel('Coefficient', fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_top_features.png'))
    plt.close()

def save_model(model, model_name, vectorizer, output_dir):
    """
    Save the trained model and vectorizer to disk.
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        vectorizer: Fitted vectorizer
        output_dir (str): Directory to save the model
    """
    print(f"Saving {model_name} to disk...")
    
    # Create model filename
    model_filename = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_model.pkl')
    
    # Save model
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Save vectorizer
    vectorizer_filename = os.path.join(output_dir, 'vectorizer.pkl')
    with open(vectorizer_filename, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model saved to {model_filename}")
    print(f"Vectorizer saved to {vectorizer_filename}")

def plot_model_comparison(accuracies, output_dir):
    """
    Plot a comparison of model accuracies.
    
    Args:
        accuracies (dict): Dictionary of model names and their accuracies
        output_dir (str): Directory to save the plot
    """
    print("Plotting model comparison...")
    
    plt.figure(figsize=(10, 6))
    
    # Sort models by accuracy
    sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    models = [item[0] for item in sorted_models]
    accs = [item[1] for item in sorted_models]
    
    # Plot bar chart
    bars = plt.bar(models, accs, color=YELLOW_PALETTE)
    
    # Add accuracy labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()

def main():
    """Main function to execute the model training pipeline."""
    # Define file paths
    input_path = '../data/processed_yelp_reviews.csv'
    output_dir = '../models'
    plot_dir = '../assets'
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load data
    df = load_data(input_path)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Extract features
    X_train_features, X_test_features, vectorizer = extract_features(X_train, X_test)
    
    # Train models
    rf_model = train_random_forest(X_train_features, y_train)
    nb_model = train_naive_bayes(X_train_features, y_train)
    lr_model = train_logistic_regression(X_train_features, y_train)
    
    # Evaluate models
    rf_accuracy = evaluate_model(rf_model, X_test_features, y_test, "Random Forest", plot_dir)
    nb_accuracy = evaluate_model(nb_model, X_test_features, y_test, "Naive Bayes", plot_dir)
    lr_accuracy = evaluate_model(lr_model, X_test_features, y_test, "Logistic Regression", plot_dir)
    
    # Get and plot top features for Logistic Regression
    top_positive, top_negative = get_top_features(vectorizer, lr_model)
    plot_top_features(top_positive, top_negative, "Logistic Regression", plot_dir)
    
    # Compare models
    accuracies = {
        "Random Forest": rf_accuracy,
        "Naive Bayes": nb_accuracy,
        "Logistic Regression": lr_accuracy
    }
    plot_model_comparison(accuracies, plot_dir)
    
    # Save best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = {"Random Forest": rf_model, "Naive Bayes": nb_model, "Logistic Regression": lr_model}[best_model_name]
    save_model(best_model, best_model_name, vectorizer, output_dir)
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()
