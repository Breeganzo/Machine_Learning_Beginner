"""
Polynomial Regression Implementation
===================================

This script demonstrates polynomial regression to capture non-linear relationships.
We'll explore how polynomial features can improve model performance for curved data.

Author: Your Learning Journey
Date: August 25, 2025
"""

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split, validation_curve  # For data splitting and validation
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # For polynomial features and scaling
from sklearn.pipeline import Pipeline  # For creating ML pipelines
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Evaluation metrics
import seaborn as sns  # For enhanced visualizations

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """
    Load the dataset and perform initial exploration
    """
    print("="*60)
    print("STEP 1: Loading and Exploring Data for Polynomial Regression")
    print("="*60)
    
    # Load the dataset
    data = pd.read_csv('dataset.csv')
    
    # Display basic information
    print("Dataset shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(data.describe())
    
    # Check for missing values
    print("\nMissing values:")
    print(data.isnull().sum())
    
    return data

def analyze_relationships(data):
    """
    Analyze relationships to identify non-linear patterns
    """
    print("\n" + "="*60)
    print("STEP 2: Analyzing Relationships for Non-linearity")
    print("="*60)
    
    # Calculate correlations
    correlations = data.corr()['price'].sort_values(ascending=False)
    print("Correlations with price:")
    for feature, corr in correlations.items():
        if feature != 'price':
            print(f"  {feature:12s}: {corr:.4f}")
    
    # Create visualization to detect non-linearity
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Relationship Analysis for Polynomial Regression', fontsize=16, fontweight='bold')
    
    # 1. Area vs Price (main focus for polynomial regression)
    axes[0, 0].scatter(data['area'], data['price'], alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('House Area (sq ft)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('Area vs Price (Check for Curves)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add a basic linear trend line
    z = np.polyfit(data['area'], data['price'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(data['area'], p(data['area']), "r--", alpha=0.8, linewidth=2, label='Linear Trend')
    axes[0, 0].legend()
    
    # 2. Age vs Price
    axes[0, 1].scatter(data['age'], data['price'], alpha=0.7, color='green')
    axes[0, 1].set_xlabel('House Age (years)')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Age vs Price')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Bedrooms vs Price
    axes[1, 0].scatter(data['bedrooms'], data['price'], alpha=0.7, color='red')
    axes[1, 0].set_xlabel('Number of Bedrooms')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Bedrooms vs Price')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals plot for linear regression (to identify non-linearity)
    # Fit simple linear regression
    X_simple = data[['area']].values
    y = data['price'].values
    model_simple = LinearRegression()
    model_simple.fit(X_simple, y)
    y_pred_simple = model_simple.predict(X_simple)
    residuals = y - y_pred_simple
    
    axes[1, 1].scatter(y_pred_simple, residuals, alpha=0.7, color='purple')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Linear Regression Residuals\n(Patterns suggest non-linearity)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze residual patterns
    print("\n🔍 Residual Analysis for Linear Regression:")
    print(f"Residual mean: {residuals.mean():.2f} (should be near 0)")
    print(f"Residual std: {residuals.std():.2f}")
    
    # Check for patterns in residuals (indication of non-linearity)
    if np.std(residuals) > np.std(y) * 0.1:
        print("⚠️  High residual variance - might benefit from polynomial features")
    else:
        print("✅ Linear relationship seems adequate")

def prepare_polynomial_features(data, degrees=[1, 2, 3, 4]):
    """
    Prepare data with different polynomial degrees
    """
    print("\n" + "="*60)
    print("STEP 3: Preparing Polynomial Features")
    print("="*60)
    
    # Use area as the main feature for polynomial regression
    X = data[['area']].values
    y = data['price'].values
    
    print(f"Original features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Create polynomial features for different degrees
    polynomial_data = {}
    
    for degree in degrees:
        print(f"\n📊 Creating polynomial features (degree {degree}):")
        
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Transform training and testing data
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        # Scale the features (important for polynomial features)
        scaler = StandardScaler()
        X_train_poly_scaled = scaler.fit_transform(X_train_poly)
        X_test_poly_scaled = scaler.transform(X_test_poly)
        
        print(f"  Original features: {X_train.shape[1]}")
        print(f"  Polynomial features: {X_train_poly.shape[1]}")
        print(f"  Feature names: {poly_features.get_feature_names_out(['area'])}")
        
        # Store data for this degree
        polynomial_data[degree] = {
            'X_train': X_train_poly_scaled,
            'X_test': X_test_poly_scaled,
            'poly_features': poly_features,
            'scaler': scaler
        }
    
    return X_train, X_test, y_train, y_test, polynomial_data

def train_polynomial_models(X_train, X_test, y_train, y_test, polynomial_data):
    """
    Train polynomial regression models with different degrees
    """
    print("\n" + "="*60)
    print("STEP 4: Training Polynomial Regression Models")
    print("="*60)
    
    results = {}
    
    # Train models for each polynomial degree
    for degree, data_dict in polynomial_data.items():
        print(f"\n🔹 Training Polynomial Regression (degree {degree}):")
        
        # Get the transformed data
        X_train_poly = data_dict['X_train']
        X_test_poly = data_dict['X_test']
        
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Testing R²: {test_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:,.0f}")
        print(f"  Testing RMSE: {test_rmse:,.0f}")
        
        # Check for overfitting
        overfitting = train_r2 - test_r2
        print(f"  Overfitting gap: {overfitting:.4f}")
        
        if overfitting > 0.1:
            print("  ⚠️  Possible overfitting detected!")
        elif overfitting > 0.05:
            print("  ⚠️  Mild overfitting")
        else:
            print("  ✅ Good generalization")
        
        # Store results
        results[degree] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'overfitting': overfitting
        }
    
    return results

def compare_polynomial_degrees(results):
    """
    Compare performance across different polynomial degrees
    """
    print("\n" + "="*60)
    print("STEP 5: Comparing Polynomial Degrees")
    print("="*60)
    
    # Create comparison table
    print("📊 Model Comparison:")
    print("-" * 80)
    print(f"{'Degree':<8} {'Train R²':<10} {'Test R²':<9} {'Train RMSE':<12} {'Test RMSE':<11} {'Overfitting':<12}")
    print("-" * 80)
    
    best_degree = None
    best_test_r2 = -np.inf
    
    for degree in sorted(results.keys()):
        result = results[degree]
        print(f"{degree:<8} {result['train_r2']:<10.4f} {result['test_r2']:<9.4f} "
              f"{result['train_rmse']:<12.0f} {result['test_rmse']:<11.0f} {result['overfitting']:<12.4f}")
        
        # Track best model (highest test R² with reasonable overfitting)
        if result['test_r2'] > best_test_r2 and result['overfitting'] < 0.15:
            best_test_r2 = result['test_r2']
            best_degree = degree
    
    print("-" * 80)
    print(f"🏆 Best performing degree: {best_degree} (Test R²: {best_test_r2:.4f})")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for degree, result in results.items():
        if degree == 1:
            print(f"  Degree {degree}: Linear baseline - simple but may underfit")
        elif degree == 2:
            print(f"  Degree {degree}: Quadratic - good balance of complexity and performance")
        elif degree == 3:
            print(f"  Degree {degree}: Cubic - more flexible but watch for overfitting")
        elif degree >= 4:
            print(f"  Degree {degree}: High-order - very flexible but prone to overfitting")
        
        if result['overfitting'] > 0.15:
            print(f"    ⚠️  Degree {degree} shows significant overfitting")
    
    return best_degree

def visualize_polynomial_regression(X_train, X_test, y_train, y_test, polynomial_data, results):
    """
    Create comprehensive visualizations for polynomial regression
    """
    print("\n" + "="*60)
    print("STEP 6: Polynomial Regression Visualization")
    print("="*60)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Main plot: All polynomial fits
    ax1 = plt.subplot(3, 3, (1, 3))
    
    # Plot training data
    plt.scatter(X_train, y_train, alpha=0.6, color='blue', s=30, label='Training Data')
    plt.scatter(X_test, y_test, alpha=0.6, color='red', s=30, label='Testing Data')
    
    # Create smooth curve for visualization
    X_smooth = np.linspace(X_train.min(), X_train.max(), 300).reshape(-1, 1)
    colors = ['green', 'orange', 'purple', 'brown']
    
    for i, (degree, data_dict) in enumerate(polynomial_data.items()):
        # Transform smooth data
        X_smooth_poly = data_dict['poly_features'].transform(X_smooth)
        X_smooth_poly_scaled = data_dict['scaler'].transform(X_smooth_poly)
        
        # Predict using the model
        model = results[degree]['model']
        y_smooth_pred = model.predict(X_smooth_poly_scaled)
        
        plt.plot(X_smooth, y_smooth_pred, color=colors[i], linewidth=2, 
                label=f'Degree {degree} (R²={results[degree]["test_r2"]:.3f})')
    
    plt.xlabel('House Area (sq ft)')
    plt.ylabel('Price ($)')
    plt.title('Polynomial Regression: Different Degrees Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2-4. Individual performance plots for degrees 1, 2, 3
    for i, degree in enumerate([1, 2, 3], start=4):
        ax = plt.subplot(3, 3, i)
        
        # Get predictions
        y_test_pred = results[degree]['y_test_pred']
        
        # Actual vs Predicted
        plt.scatter(y_test, y_test_pred, alpha=0.7, color=colors[degree-1])
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'Degree {degree}: Actual vs Predicted\n(R²={results[degree]["test_r2"]:.3f})')
        plt.grid(True, alpha=0.3)
    
    # 5. Learning curves comparison
    ax5 = plt.subplot(3, 3, 7)
    degrees = list(results.keys())
    train_scores = [results[d]['train_r2'] for d in degrees]
    test_scores = [results[d]['test_r2'] for d in degrees]
    
    plt.plot(degrees, train_scores, 'bo-', linewidth=2, label='Training R²')
    plt.plot(degrees, test_scores, 'ro-', linewidth=2, label='Testing R²')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('Learning Curves: R² vs Polynomial Degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Overfitting analysis
    ax6 = plt.subplot(3, 3, 8)
    overfitting_gaps = [results[d]['overfitting'] for d in degrees]
    
    bars = plt.bar(degrees, overfitting_gaps, alpha=0.7, 
                   color=['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red' 
                          for gap in overfitting_gaps])
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Mild Overfitting')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Significant Overfitting')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Overfitting Gap (Train R² - Test R²)')
    plt.title('Overfitting Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. RMSE comparison
    ax7 = plt.subplot(3, 3, 9)
    train_rmse = [results[d]['train_rmse'] for d in degrees]
    test_rmse = [results[d]['test_rmse'] for d in degrees]
    
    plt.plot(degrees, train_rmse, 'bo-', linewidth=2, label='Training RMSE')
    plt.plot(degrees, test_rmse, 'ro-', linewidth=2, label='Testing RMSE')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Polynomial Degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_overfitting(X_train, y_train):
    """
    Demonstrate overfitting with very high degree polynomials
    """
    print("\n" + "="*60)
    print("STEP 7: Demonstrating Overfitting with High Degrees")
    print("="*60)
    
    # Create validation curve for degrees 1 to 10
    degrees = range(1, 11)
    
    # Use Pipeline for clean implementation
    pipe = Pipeline([
        ('poly', PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])
    
    # Calculate validation curve
    train_scores, val_scores = validation_curve(
        pipe, X_train, y_train,
        param_name='poly__degree',
        param_range=degrees,
        cv=5,  # 5-fold cross-validation
        scoring='r2'
    )
    
    # Calculate means and standard deviations
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Plot validation curve
    plt.figure(figsize=(12, 8))
    
    plt.plot(degrees, train_mean, 'bo-', linewidth=2, label='Training Score')
    plt.fill_between(degrees, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(degrees, val_mean, 'ro-', linewidth=2, label='Validation Score')
    plt.fill_between(degrees, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('Validation Curve: Demonstrating Overfitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Analysis
    print("📊 Overfitting Analysis:")
    print("-" * 50)
    print(f"{'Degree':<8} {'Train R²':<10} {'Val R²':<8} {'Gap':<8}")
    print("-" * 50)
    
    best_val_score = -np.inf
    best_degree = 1
    
    for i, degree in enumerate(degrees):
        gap = train_mean[i] - val_mean[i]
        print(f"{degree:<8} {train_mean[i]:<10.4f} {val_mean[i]:<8.4f} {gap:<8.4f}")
        
        if val_mean[i] > best_val_score:
            best_val_score = val_mean[i]
            best_degree = degree
    
    print("-" * 50)
    print(f"🏆 Best degree by validation: {best_degree} (R²: {best_val_score:.4f})")
    
    # Recommendations
    print(f"\n💡 Key Observations:")
    print(f"- Training score generally increases with degree")
    print(f"- Validation score peaks and then decreases (overfitting)")
    print(f"- Gap between training and validation indicates overfitting")
    print(f"- Choose degree that maximizes validation performance")

def polynomial_feature_analysis(polynomial_data, results, best_degree):
    """
    Analyze the polynomial features and their coefficients
    """
    print("\n" + "="*60)
    print("STEP 8: Polynomial Feature Analysis")
    print("="*60)
    
    # Analyze the best performing model
    print(f"🔍 Analyzing Degree {best_degree} Polynomial Model:")
    
    # Get feature names and coefficients
    poly_features = polynomial_data[best_degree]['poly_features']
    model = results[best_degree]['model']
    feature_names = poly_features.get_feature_names_out(['area'])
    coefficients = model.coef_
    
    print(f"\nPolynomial equation components:")
    print(f"Intercept: {model.intercept_:.2f}")
    
    for name, coef in zip(feature_names, coefficients):
        print(f"{name:12s}: {coef:10.2f}")
    
    # Feature importance (absolute coefficient values)
    feature_importance = np.abs(coefficients)
    importance_order = np.argsort(feature_importance)[::-1]
    
    print(f"\n📊 Feature Importance (by coefficient magnitude):")
    for i, idx in enumerate(importance_order, 1):
        print(f"{i}. {feature_names[idx]:12s}: {feature_importance[idx]:8.2f}")
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(feature_names)), feature_importance, alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Magnitude')
    plt.title(f'Feature Importance (Degree {best_degree})')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(feature_names)), coefficients, alpha=0.7, 
            color=['red' if c < 0 else 'blue' for c in coefficients])
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title(f'Coefficient Values (Degree {best_degree})')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the complete polynomial regression workflow
    """
    print("🏠 POLYNOMIAL REGRESSION: NON-LINEAR HOUSE PRICE PREDICTION")
    print("📈 Exploring polynomial features to capture non-linear relationships")
    print("="*75)
    
    try:
        # Execute the complete workflow
        data = load_and_explore_data()
        analyze_relationships(data)
        
        X_train, X_test, y_train, y_test, polynomial_data = prepare_polynomial_features(data)
        
        results = train_polynomial_models(X_train, X_test, y_train, y_test, polynomial_data)
        
        best_degree = compare_polynomial_degrees(results)
        
        visualize_polynomial_regression(X_train, X_test, y_train, y_test, polynomial_data, results)
        
        demonstrate_overfitting(X_train, y_train)
        
        polynomial_feature_analysis(polynomial_data, results, best_degree)
        
        print("\n" + "="*75)
        print("✅ Polynomial Regression analysis completed successfully!")
        print("📊 You learned how polynomial features can capture non-linear relationships!")
        print("⚠️  Remember: Higher degree ≠ Better performance (overfitting risk)")
        print("="*75)
        
    except FileNotFoundError:
        print("❌ Error: dataset.csv not found!")
        print("Make sure the dataset file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

# Execute the main function when script is run directly
if __name__ == "__main__":
    main()
