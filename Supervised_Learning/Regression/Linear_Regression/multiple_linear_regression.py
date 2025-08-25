"""
Multiple Linear Regression Implementation
=======================================

This script demonstrates multiple linear regression using multiple features to predict house prices.
We'll use area, bedrooms, and age to predict the price.

Author: Your Learning Journey
Date: August 25, 2025
"""

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LinearRegression     # Linear regression model
from sklearn.preprocessing import StandardScaler     # For feature scaling
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Evaluation metrics
import seaborn as sns  # For enhanced visualizations

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """
    Load the dataset and perform comprehensive exploration
    """
    print("="*50)
    print("STEP 1: Loading and Exploring Data")
    print("="*50)
    
    # Load the dataset
    data = pd.read_csv('dataset.csv')
    
    # Display basic information about the dataset
    print("Dataset shape:", data.shape)
    print("\nDataset columns:", list(data.columns))
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

def analyze_correlations(data):
    """
    Analyze correlations between features and target variable
    """
    print("\n" + "="*50)
    print("STEP 2: Correlation Analysis")
    print("="*50)
    
    # Calculate correlation matrix
    correlation_matrix = data.corr()
    
    print("Correlation Matrix:")
    print(correlation_matrix)
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of All Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Analyze correlations with target variable (price)
    price_correlations = correlation_matrix['price'].sort_values(ascending=False)
    print(f"\nCorrelations with Price (sorted):")
    for feature, corr in price_correlations.items():
        if feature != 'price':
            print(f"{feature:12s}: {corr:6.3f}")
    
    # Interpretation
    print(f"\nCorrelation Interpretation:")
    for feature, corr in price_correlations.items():
        if feature != 'price':
            if abs(corr) > 0.7:
                strength = "Strong"
            elif abs(corr) > 0.5:
                strength = "Moderate"
            elif abs(corr) > 0.3:
                strength = "Weak"
            else:
                strength = "Very weak"
            
            direction = "positive" if corr > 0 else "negative"
            print(f"- {feature}: {strength} {direction} correlation")

def visualize_multiple_relationships(data):
    """
    Create comprehensive visualizations for multiple features
    """
    print("\n" + "="*50)
    print("STEP 3: Multi-Feature Visualization")
    print("="*50)
    
    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multiple Linear Regression: Feature Analysis', fontsize=16, fontweight='bold')
    
    # 1. Area vs Price
    axes[0, 0].scatter(data['area'], data['price'], alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('House Area (sq ft)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('Area vs Price')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Bedrooms vs Price
    axes[0, 1].scatter(data['bedrooms'], data['price'], alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Number of Bedrooms')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Bedrooms vs Price')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Age vs Price
    axes[0, 2].scatter(data['age'], data['price'], alpha=0.7, color='red')
    axes[0, 2].set_xlabel('House Age (years)')
    axes[0, 2].set_ylabel('Price ($)')
    axes[0, 2].set_title('Age vs Price')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Distribution of Area
    axes[1, 0].hist(data['area'], bins=15, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel('House Area (sq ft)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Area')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Distribution of Bedrooms
    bedroom_counts = data['bedrooms'].value_counts().sort_index()
    axes[1, 1].bar(bedroom_counts.index, bedroom_counts.values, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_xlabel('Number of Bedrooms')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Bedrooms')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Distribution of Age
    axes[1, 2].hist(data['age'], bins=15, alpha=0.7, color='red', edgecolor='black')
    axes[1, 2].set_xlabel('House Age (years)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Distribution of Age')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def prepare_multiple_features(data):
    """
    Prepare data for multiple linear regression
    """
    print("\n" + "="*50)
    print("STEP 4: Data Preparation for Multiple Features")
    print("="*50)
    
    # Select multiple features for the model
    feature_columns = ['area', 'bedrooms', 'age']  # Multiple independent variables
    X = data[feature_columns].values  # Features matrix
    y = data['price'].values          # Target variable
    
    print(f"Selected features: {feature_columns}")
    print(f"Features shape: {X.shape}")  # Should be (n_samples, n_features)
    print(f"Target shape: {y.shape}")    # Should be (n_samples,)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,     # 20% for testing
        random_state=42    # For reproducible results
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Feature Scaling (important for multiple features)
    print(f"\nApplying Feature Scaling...")
    scaler = StandardScaler()
    
    # Fit the scaler on training data and transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    print("Original vs Scaled feature statistics:")
    
    # Show scaling effect
    feature_names = feature_columns
    for i, feature in enumerate(feature_names):
        print(f"\n{feature}:")
        print(f"  Original - Mean: {X_train[:, i].mean():.2f}, Std: {X_train[:, i].std():.2f}")
        print(f"  Scaled   - Mean: {X_train_scaled[:, i].mean():.2f}, Std: {X_train_scaled[:, i].std():.2f}")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_columns

def train_multiple_regression(X_train_scaled, y_train, feature_columns):
    """
    Train the multiple linear regression model
    """
    print("\n" + "="*50)
    print("STEP 5: Multiple Linear Regression Training")
    print("="*50)
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Extract the learned parameters
    coefficients = model.coef_      # Coefficients for each feature
    intercept = model.intercept_    # Intercept (bias)
    
    print("Model trained successfully!")
    print(f"Intercept: {intercept:.2f}")
    print("Coefficients:")
    
    # Display equation
    equation_parts = [f"{intercept:.2f}"]
    for i, (feature, coef) in enumerate(zip(feature_columns, coefficients)):
        sign = "+" if coef >= 0 else "-"
        equation_parts.append(f" {sign} {abs(coef):.2f} * {feature}")
    
    equation = "Price = " + "".join(equation_parts)
    print(f"\nLearned equation (with scaled features):")
    print(equation)
    
    # Feature importance analysis
    print(f"\nFeature Importance Analysis:")
    feature_importance = list(zip(feature_columns, abs(coefficients)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {feature:12s}: {importance:.2f} (coefficient magnitude)")
    
    # Interpretation of coefficients
    print(f"\nCoefficient Interpretation:")
    for feature, coef in zip(feature_columns, coefficients):
        if coef > 0:
            print(f"- {feature}: Positive impact (+{coef:.2f}) - increases price")
        else:
            print(f"- {feature}: Negative impact ({coef:.2f}) - decreases price")
    
    return model

def make_multiple_predictions(model, X_test_scaled):
    """
    Make predictions using the multiple linear regression model
    """
    print("\n" + "="*50)
    print("STEP 6: Making Predictions with Multiple Features")
    print("="*50)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    print("Predictions completed!")
    print(f"Number of predictions: {len(y_pred)}")
    
    return y_pred

def evaluate_multiple_model(y_test, y_pred, X_test, feature_columns):
    """
    Comprehensive evaluation of the multiple linear regression model
    """
    print("\n" + "="*50)
    print("STEP 7: Model Evaluation")
    print("="*50)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Model interpretation
    print(f"\nModel Performance Interpretation:")
    print(f"- The model explains {r2*100:.1f}% of the variance in house prices")
    print(f"- On average, predictions are off by ${rmse:,.0f}")
    print(f"- Mean absolute error: ${mae:,.0f}")
    
    # Calculate percentage error
    mean_actual = np.mean(y_test)
    percentage_error = (rmse / mean_actual) * 100
    print(f"- Average percentage error: {percentage_error:.1f}%")
    
    # Performance rating
    if r2 > 0.8:
        rating = "Excellent"
    elif r2 > 0.6:
        rating = "Good"
    elif r2 > 0.4:
        rating = "Fair"
    else:
        rating = "Poor"
    
    print(f"- Overall model performance: {rating}")
    
    return mse, rmse, mae, r2

def visualize_multiple_results(X_test, y_test, y_pred, model, feature_columns):
    """
    Create comprehensive visualizations for multiple regression results
    """
    print("\n" + "="*50)
    print("STEP 8: Results Visualization")
    print("="*50)
    
    # Create a comprehensive results figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multiple Linear Regression Results', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted scatter plot
    axes[0, 0].scatter(y_test, y_pred, alpha=0.7, color='blue')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    axes[0, 0].set_xlabel('Actual Prices')
    axes[0, 0].set_ylabel('Predicted Prices')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.7, color='purple')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Prices')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals histogram
    axes[0, 2].hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Residuals')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Feature importance (coefficient magnitudes)
    coefficients = model.coef_
    coef_abs = np.abs(coefficients)
    axes[1, 0].bar(feature_columns, coef_abs, alpha=0.7, color=['blue', 'green', 'red'])
    axes[1, 0].set_xlabel('Features')
    axes[1, 0].set_ylabel('Coefficient Magnitude')
    axes[1, 0].set_title('Feature Importance')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Prediction errors distribution
    errors = np.abs(residuals)
    axes[1, 1].hist(errors, bins=15, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Absolute Errors')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Q-Q plot for residuals normality check
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot (Residuals Normality)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed predictions
    print("\nDetailed Prediction Analysis:")
    print("-" * 80)
    print(f"{'Area':>6} {'Beds':>5} {'Age':>4} {'Actual':>8} {'Predicted':>10} {'Error':>8} {'Error%':>7}")
    print("-" * 80)
    
    for i in range(min(10, len(X_test))):
        area, bedrooms, age = X_test[i]
        actual = y_test[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100
        
        print(f"{area:6.0f} {bedrooms:5.0f} {age:4.0f} ${actual:7.0f} ${predicted:9.0f} ${error:7.0f} {error_pct:6.1f}%")

def compare_with_simple_regression(data):
    """
    Compare multiple regression performance with simple regression
    """
    print("\n" + "="*50)
    print("STEP 9: Comparison with Simple Linear Regression")
    print("="*50)
    
    # Simple regression (area only)
    X_simple = data[['area']].values
    y = data['price'].values
    
    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
        X_simple, y, test_size=0.2, random_state=42
    )
    
    # Train simple model
    simple_model = LinearRegression()
    simple_model.fit(X_train_simple, y_train_simple)
    y_pred_simple = simple_model.predict(X_test_simple)
    
    # Calculate metrics for simple model
    r2_simple = r2_score(y_test_simple, y_pred_simple)
    rmse_simple = np.sqrt(mean_squared_error(y_test_simple, y_pred_simple))
    
    # Multiple regression (already calculated)
    X_multiple = data[['area', 'bedrooms', 'age']].values
    X_train_mult, X_test_mult, y_train_mult, y_test_mult = train_test_split(
        X_multiple, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_mult_scaled = scaler.fit_transform(X_train_mult)
    X_test_mult_scaled = scaler.transform(X_test_mult)
    
    multiple_model = LinearRegression()
    multiple_model.fit(X_train_mult_scaled, y_train_mult)
    y_pred_mult = multiple_model.predict(X_test_mult_scaled)
    
    r2_multiple = r2_score(y_test_mult, y_pred_mult)
    rmse_multiple = np.sqrt(mean_squared_error(y_test_mult, y_pred_mult))
    
    # Comparison
    print("Performance Comparison:")
    print(f"{'Metric':<20} {'Simple Regression':<18} {'Multiple Regression':<20} {'Improvement':<12}")
    print("-" * 75)
    print(f"{'R² Score':<20} {r2_simple:<18.4f} {r2_multiple:<20.4f} {r2_multiple-r2_simple:+.4f}")
    print(f"{'RMSE':<20} {rmse_simple:<18.0f} {rmse_multiple:<20.0f} {rmse_simple-rmse_multiple:+.0f}")
    
    improvement_r2 = ((r2_multiple - r2_simple) / r2_simple) * 100
    improvement_rmse = ((rmse_simple - rmse_multiple) / rmse_simple) * 100
    
    print(f"\nImprovement Summary:")
    print(f"- R² improved by {improvement_r2:.1f}%")
    print(f"- RMSE improved by {improvement_rmse:.1f}%")
    
    if improvement_r2 > 5:
        print("✅ Multiple regression shows significant improvement!")
    else:
        print("⚠️  Multiple regression shows modest improvement.")

def main():
    """
    Main function to run the complete multiple linear regression workflow
    """
    print("🏠 MULTIPLE LINEAR REGRESSION: HOUSE PRICE PREDICTION")
    print("📊 Using Area, Bedrooms, and Age to Predict Price")
    print("="*65)
    
    try:
        # Execute the complete workflow
        data = load_and_explore_data()
        analyze_correlations(data)
        visualize_multiple_relationships(data)
        
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_columns = prepare_multiple_features(data)
        
        model = train_multiple_regression(X_train_scaled, y_train, feature_columns)
        y_pred = make_multiple_predictions(model, X_test_scaled)
        
        mse, rmse, mae, r2 = evaluate_multiple_model(y_test, y_pred, X_test, feature_columns)
        visualize_multiple_results(X_test, y_test, y_pred, model, feature_columns)
        
        compare_with_simple_regression(data)
        
        print("\n" + "="*65)
        print("✅ Multiple Linear Regression completed successfully!")
        print("📈 Multiple features provide better predictions than single feature!")
        print("="*65)
        
    except FileNotFoundError:
        print("❌ Error: dataset.csv not found!")
        print("Make sure the dataset file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

# Execute the main function when script is run directly
if __name__ == "__main__":
    main()
