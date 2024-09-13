import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the DataFrame by handling missing values and converting data types.
    """
    df = df.dropna()  # Drop rows with missing values
    return df

def visualize_data(df):
    """
    Visualize the data using a simple histogram.
    """
    df.hist(figsize=(10, 6))
    plt.suptitle('Feature Distributions')
    plt.show()

def train_model(df):
    """
    Train a RandomForest model and return its accuracy.
    """
    # Assume the target column is named 'target' and features are all other columns
    X = df.drop('target', axis=1)  # Features
    y = df['target']  # Target variable
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train the RandomForest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return accuracy

def main():
    """
    Main function to orchestrate data loading, preprocessing, visualization, and modeling.
    """
    file_path = 'data.csv'  # Replace with your CSV file path
    df = load_data(file_path)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Visualize the data
    visualize_data(df)
    
    # Train the model and print accuracy
    accuracy = train_model(df)
    print(f'Model Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    main()
