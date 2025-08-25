"""
Simple Linear Regression Implementation
=====================================

This script demonstrates simple linear regression using one feature to predict house prices.
We'll use the house area to predict the price.

Author: Your Learning Journey
Date: August 25, 2025
"""

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LinearRegression     # Linear regression model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Evaluation metrics
import seaborn as sns  # For enhanced visualizations

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """
    Load the dataset and perform initial exploration
    """
    print("="*50)
    print("STEP 1: Loading and Exploring Data")
    print("="*50)
    
    # Load the dataset
    data = pd.read_csv('dataset.csv')
    
    # Display basic information about the dataset
    print("Dataset shape:", data.shape)  # Number of rows and columns
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(data.describe())
    
    # Check data types
    print("\nData Types:")
    print(data.dtypes)
    
    return data

def visualize_data(data):
    """
    Create visualizations to understand the data better
    """
    print("\n" + "="*50)
    print("STEP 2: Data Visualization")
    print("="*50)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('House Price Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Area vs Price (our main relationship)
    axes[0, 0].scatter(data['area'], data['price'], alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('House Area (sq ft)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('Area vs Price (Simple Linear Relationship)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution of house areas
    axes[0, 1].hist(data['area'], bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('House Area (sq ft)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of House Areas')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of house prices
    axes[1, 0].hist(data['price'], bins=15, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Price ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of House Prices')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot of prices
    axes[1, 1].boxplot(data['price'])
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].set_title('Box Plot of House Prices')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation
    correlation = data['area'].corr(data['price'])
    print(f"Correlation between Area and Price: {correlation:.4f}")
    
    if correlation > 0.7:
        print("Strong positive correlation - Good for linear regression!")
    elif correlation > 0.5:
        print("Moderate positive correlation - Linear regression should work well.")
    else:
        print("Weak correlation - Linear regression might not be the best choice.")

def prepare_data(data):
    """
    Prepare data for machine learning
    """
    print("\n" + "="*50)
    print("STEP 3: Data Preparation")
    print("="*50)
    
    # For simple linear regression, we use only one feature (area)
    X = data[['area']].values  # Features (independent variable) - note the double brackets for 2D array
    y = data['price'].values   # Target (dependent variable)
    
    print(f"Features shape: {X.shape}")  # Should be (n_samples, 1)
    print(f"Target shape: {y.shape}")    # Should be (n_samples,)
    
    # Split the data into training and testing sets
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,     # 20% for testing
        random_state=42    # For reproducible results
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train the linear regression model
    """
    print("\n" + "="*50)
    print("STEP 4: Model Training")
    print("="*50)
    
    # Create a linear regression model
    model = LinearRegression()
    
    # Train the model using the training data
    # The model learns the relationship: price = slope * area + intercept
    model.fit(X_train, y_train)
    
    # Extract the learned parameters
    slope = model.coef_[0]        # Coefficient (slope)
    intercept = model.intercept_  # Intercept (bias)
    
    print(f"Model trained successfully!")
    print(f"Learned equation: Price = {slope:.2f} * Area + {intercept:.2f}")
    print(f"Slope (coefficient): {slope:.2f}")
    print(f"Intercept: {intercept:.2f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"- For every 1 sq ft increase in area, price increases by ${slope:.2f}")
    print(f"- A house with 0 sq ft would theoretically cost ${intercept:.2f} (intercept)")
    
    return model

def make_predictions(model, X_test):
    """
    Make predictions using the trained model
    """
    print("\n" + "="*50)
    print("STEP 5: Making Predictions")
    print("="*50)
    
    # Use the trained model to make predictions on test data
    y_pred = model.predict(X_test)
    
    print("Predictions made on test set!")
    print(f"Number of predictions: {len(y_pred)}")
    
    return y_pred

def evaluate_model(y_test, y_pred):
    """
    Evaluate the model performance using various metrics
    """
    print("\n" + "="*50)
    print("STEP 6: Model Evaluation")
    print("="*50)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)           # Mean Squared Error
    rmse = np.sqrt(mse)                                # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)          # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)                      # R-squared score
    
    print("Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Interpret R-squared
    print(f"\nR-squared Interpretation:")
    if r2 > 0.8:
        print(f"Excellent! The model explains {r2*100:.1f}% of the variance in house prices.")
    elif r2 > 0.6:
        print(f"Good! The model explains {r2*100:.1f}% of the variance in house prices.")
    elif r2 > 0.4:
        print(f"Fair. The model explains {r2*100:.1f}% of the variance in house prices.")
    else:
        print(f"Poor. The model only explains {r2*100:.1f}% of the variance in house prices.")
    
    # Calculate percentage error
    mean_actual = np.mean(y_test)
    percentage_error = (rmse / mean_actual) * 100
    print(f"Average percentage error: {percentage_error:.1f}%")
    
    return mse, rmse, mae, r2

def visualize_results(X_test, y_test, y_pred, model):
    """
    Create visualizations to show model performance
    """
    print("\n" + "="*50)
    print("STEP 7: Results Visualization")
    print("="*50)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Simple Linear Regression Results', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_test, y_pred, alpha=0.7, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Prices')
    axes[0, 0].set_ylabel('Predicted Prices')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Regression line with test data
    axes[0, 1].scatter(X_test, y_test, alpha=0.7, color='blue', label='Actual')
    axes[0, 1].scatter(X_test, y_pred, alpha=0.7, color='red', label='Predicted')
    
    # Plot the regression line
    X_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    axes[0, 1].plot(X_line, y_line, 'g-', linewidth=2, label='Regression Line')
    
    axes[0, 1].set_xlabel('House Area (sq ft)')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Linear Regression Fit')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals plot
    residuals = y_test - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.7, color='purple')
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Prices')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals histogram
    axes[1, 1].hist(residuals, bins=10, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some example predictions
    print("\nSample Predictions:")
    print("-" * 40)
    for i in range(min(5, len(X_test))):
        area = X_test[i][0]
        actual = y_test[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        print(f"Area: {area:4.0f} sq ft | Actual: ${actual:6.0f} | Predicted: ${predicted:6.0f} | Error: ${error:5.0f}")

def main():
    """
    Main function to run the complete simple linear regression workflow
    """
    print("🏠 SIMPLE LINEAR REGRESSION: HOUSE PRICE PREDICTION")
    print("📊 Using House Area to Predict Price")
    print("="*60)
    
    try:
        # Step 1: Load and explore data
        data = load_and_explore_data()
        
        # Step 2: Visualize data
        visualize_data(data)
        
        # Step 3: Prepare data
        X_train, X_test, y_train, y_test = prepare_data(data)
        
        # Step 4: Train model
        model = train_model(X_train, y_train)
        
        # Step 5: Make predictions
        y_pred = make_predictions(model, X_test)
        
        # Step 6: Evaluate model
        mse, rmse, mae, r2 = evaluate_model(y_test, y_pred)
        
        # Step 7: Visualize results
        visualize_results(X_test, y_test, y_pred, model)
        
        print("\n" + "="*60)
        print("✅ Simple Linear Regression completed successfully!")
        print("📈 Check the visualizations to understand the model performance.")
        print("="*60)
        
    except FileNotFoundError:
        print("❌ Error: dataset.csv not found!")
        print("Make sure the dataset file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

# Execute the main function when script is run directly
if __name__ == "__main__":
    main()
