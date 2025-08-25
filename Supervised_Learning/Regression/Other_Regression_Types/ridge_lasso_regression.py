"""
Ridge and Lasso Regression Implementation
========================================

This script demonstrates regularized regression techniques (Ridge and Lasso) 
to prevent overfitting and perform feature selection.

Author: Your Learning Journey
Date: August 25, 2025
"""

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve  # ML tools
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet  # Regression models
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # Evaluation metrics
import seaborn as sns  # For enhanced visualizations

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """
    Load the dataset and add some correlated features to demonstrate regularization
    """
    print("="*70)
    print("STEP 1: Loading and Preparing Data for Regularization")
    print("="*70)
    
    # Load the dataset
    data = pd.read_csv('dataset.csv')
    
    print("Original dataset shape:", data.shape)
    print("\nOriginal columns:", list(data.columns))
    
    # Add some engineered features to demonstrate regularization
    # These will create multicollinearity that regularization can handle
    data['area_squared'] = data['area'] ** 2  # Non-linear feature
    data['area_sqrt'] = np.sqrt(data['area'])  # Another non-linear feature
    data['bedroom_area_ratio'] = data['bedrooms'] / data['area']  # Ratio feature
    data['age_squared'] = data['age'] ** 2  # Age squared
    data['area_age_interaction'] = data['area'] * data['age']  # Interaction feature
    data['luxury_score'] = (data['area'] / 1000) * (5 - data['bedrooms']) * (20 - data['age'])  # Complex feature
    
    # Add some noise features to demonstrate feature selection
    np.random.seed(42)
    data['noise_1'] = np.random.normal(0, 1, len(data))  # Pure noise
    data['noise_2'] = np.random.normal(0, 1, len(data))  # Pure noise
    data['semi_corr'] = data['price'] * 0.1 + np.random.normal(0, 10000, len(data))  # Weakly correlated noise
    
    print(f"Enhanced dataset shape: {data.shape}")
    print(f"Added features: {[col for col in data.columns if col not in ['area', 'bedrooms', 'age', 'price']]}")
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(data.describe())
    
    return data

def analyze_multicollinearity(data):
    """
    Analyze multicollinearity in the features
    """
    print("\n" + "="*70)
    print("STEP 2: Analyzing Multicollinearity")
    print("="*70)
    
    # Select feature columns (exclude target)
    feature_columns = [col for col in data.columns if col != 'price']
    X = data[feature_columns]
    
    # Calculate correlation matrix
    correlation_matrix = X.corr()
    
    print("Feature correlation analysis:")
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.7:  # High correlation threshold
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                      correlation_matrix.columns[j], corr))
    
    if high_corr_pairs:
        print(f"\n⚠️  High correlation pairs (>0.7):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  {feat1} ↔ {feat2}: {corr:.3f}")
    else:
        print("\n✅ No highly correlated features found")
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Analyze correlations with target
    target_correlations = data.corr()['price'].sort_values(ascending=False)
    print(f"\nCorrelations with target (price):")
    for feature, corr in target_correlations.items():
        if feature != 'price':
            print(f"  {feature:20s}: {corr:6.3f}")
    
    return feature_columns

def prepare_for_regularization(data, feature_columns):
    """
    Prepare data for regularized regression
    """
    print("\n" + "="*70)
    print("STEP 3: Data Preparation for Regularization")
    print("="*70)
    
    # Separate features and target
    X = data[feature_columns].values
    y = data['price'].values
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Feature scaling (CRUCIAL for regularized regression)
    print(f"\n🔧 Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully!")
    
    # Show scaling effect
    print(f"\nScaling effect (first 3 features):")
    for i in range(min(3, X_train.shape[1])):
        print(f"  Feature {i}:")
        print(f"    Original - Mean: {X_train[:, i].mean():.2f}, Std: {X_train[:, i].std():.2f}")
        print(f"    Scaled   - Mean: {X_train_scaled[:, i].mean():.2f}, Std: {X_train_scaled[:, i].std():.2f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_baseline_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train baseline linear regression model (no regularization)
    """
    print("\n" + "="*70)
    print("STEP 4: Training Baseline Linear Regression")
    print("="*70)
    
    # Train linear regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = linear_model.predict(X_train_scaled)
    y_test_pred = linear_model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("📊 Baseline Linear Regression Results:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  Training RMSE: {train_rmse:,.0f}")
    print(f"  Testing RMSE: {test_rmse:,.0f}")
    print(f"  Overfitting gap: {train_r2 - test_r2:.4f}")
    
    if train_r2 - test_r2 > 0.1:
        print("  ⚠️  Significant overfitting detected - regularization needed!")
    else:
        print("  ✅ Good generalization")
    
    return {
        'model': linear_model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

def train_ridge_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train Ridge regression with hyperparameter tuning
    """
    print("\n" + "="*70)
    print("STEP 5: Training Ridge Regression (L2 Regularization)")
    print("="*70)
    
    # Define alpha values to test
    alphas = np.logspace(-4, 2, 50)  # From 0.0001 to 100
    
    print(f"🔍 Testing {len(alphas)} alpha values for Ridge regression...")
    
    # Use GridSearchCV for hyperparameter tuning
    ridge = Ridge()
    grid_search = GridSearchCV(
        ridge, 
        param_grid={'alpha': alphas},
        cv=5,  # 5-fold cross-validation
        scoring='r2',
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    best_ridge = grid_search.best_estimator_
    best_alpha = grid_search.best_params_['alpha']
    
    print(f"✅ Best alpha for Ridge: {best_alpha:.6f}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    # Make predictions with best model
    y_train_pred = best_ridge.predict(X_train_scaled)
    y_test_pred = best_ridge.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n📊 Ridge Regression Results:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  Training RMSE: {train_rmse:,.0f}")
    print(f"  Testing RMSE: {test_rmse:,.0f}")
    print(f"  Overfitting gap: {train_r2 - test_r2:.4f}")
    
    return {
        'model': best_ridge,
        'best_alpha': best_alpha,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'grid_search': grid_search
    }

def train_lasso_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train Lasso regression with hyperparameter tuning
    """
    print("\n" + "="*70)
    print("STEP 6: Training Lasso Regression (L1 Regularization)")
    print("="*70)
    
    # Define alpha values to test
    alphas = np.logspace(-4, 2, 50)  # From 0.0001 to 100
    
    print(f"🔍 Testing {len(alphas)} alpha values for Lasso regression...")
    
    # Use GridSearchCV for hyperparameter tuning
    lasso = Lasso(max_iter=2000)  # Increase max_iter for convergence
    grid_search = GridSearchCV(
        lasso, 
        param_grid={'alpha': alphas},
        cv=5,  # 5-fold cross-validation
        scoring='r2',
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    best_lasso = grid_search.best_estimator_
    best_alpha = grid_search.best_params_['alpha']
    
    print(f"✅ Best alpha for Lasso: {best_alpha:.6f}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    # Analyze feature selection
    selected_features = np.sum(best_lasso.coef_ != 0)
    total_features = len(best_lasso.coef_)
    
    print(f"🎯 Feature Selection:")
    print(f"  Total features: {total_features}")
    print(f"  Selected features: {selected_features}")
    print(f"  Eliminated features: {total_features - selected_features}")
    print(f"  Selection ratio: {selected_features/total_features:.2%}")
    
    # Make predictions with best model
    y_train_pred = best_lasso.predict(X_train_scaled)
    y_test_pred = best_lasso.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n📊 Lasso Regression Results:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  Training RMSE: {train_rmse:,.0f}")
    print(f"  Testing RMSE: {test_rmse:,.0f}")
    print(f"  Overfitting gap: {train_r2 - test_r2:.4f}")
    
    return {
        'model': best_lasso,
        'best_alpha': best_alpha,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'selected_features': selected_features,
        'grid_search': grid_search
    }

def train_elastic_net(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train Elastic Net regression (combines Ridge and Lasso)
    """
    print("\n" + "="*70)
    print("STEP 7: Training Elastic Net Regression (L1 + L2)")
    print("="*70)
    
    # Define parameter grid
    param_grid = {
        'alpha': np.logspace(-4, 1, 20),  # Regularization strength
        'l1_ratio': np.linspace(0.1, 0.9, 9)  # Mix of L1 and L2 (0=Ridge, 1=Lasso)
    }
    
    print(f"🔍 Testing {len(param_grid['alpha']) * len(param_grid['l1_ratio'])} parameter combinations...")
    
    # Use GridSearchCV
    elastic_net = ElasticNet(max_iter=2000)
    grid_search = GridSearchCV(
        elastic_net,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    best_elastic = grid_search.best_estimator_
    best_alpha = grid_search.best_params_['alpha']
    best_l1_ratio = grid_search.best_params_['l1_ratio']
    
    print(f"✅ Best alpha: {best_alpha:.6f}")
    print(f"✅ Best l1_ratio: {best_l1_ratio:.3f}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    # Interpret l1_ratio
    if best_l1_ratio < 0.3:
        ratio_interpretation = "More Ridge-like (L2 penalty dominant)"
    elif best_l1_ratio > 0.7:
        ratio_interpretation = "More Lasso-like (L1 penalty dominant)"
    else:
        ratio_interpretation = "Balanced mix of L1 and L2 penalties"
    
    print(f"📊 L1_ratio interpretation: {ratio_interpretation}")
    
    # Analyze feature selection
    selected_features = np.sum(best_elastic.coef_ != 0)
    total_features = len(best_elastic.coef_)
    
    print(f"🎯 Feature Selection:")
    print(f"  Selected features: {selected_features}/{total_features}")
    print(f"  Selection ratio: {selected_features/total_features:.2%}")
    
    # Make predictions
    y_train_pred = best_elastic.predict(X_train_scaled)
    y_test_pred = best_elastic.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n📊 Elastic Net Results:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  Training RMSE: {train_rmse:,.0f}")
    print(f"  Testing RMSE: {test_rmse:,.0f}")
    print(f"  Overfitting gap: {train_r2 - test_r2:.4f}")
    
    return {
        'model': best_elastic,
        'best_alpha': best_alpha,
        'best_l1_ratio': best_l1_ratio,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'selected_features': selected_features,
        'grid_search': grid_search
    }

def compare_all_models(linear_results, ridge_results, lasso_results, elastic_results):
    """
    Compare all regression models
    """
    print("\n" + "="*70)
    print("STEP 8: Comparing All Regression Models")
    print("="*70)
    
    # Create comparison table
    models = ['Linear', 'Ridge', 'Lasso', 'Elastic Net']
    results = [linear_results, ridge_results, lasso_results, elastic_results]
    
    print("📊 Model Comparison:")
    print("-" * 90)
    print(f"{'Model':<12} {'Train R²':<9} {'Test R²':<8} {'Train RMSE':<12} {'Test RMSE':<11} {'Overfitting':<12}")
    print("-" * 90)
    
    best_model = None
    best_test_r2 = -np.inf
    
    for model_name, result in zip(models, results):
        overfitting = result['train_r2'] - result['test_r2']
        
        print(f"{model_name:<12} {result['train_r2']:<9.4f} {result['test_r2']:<8.4f} "
              f"{result['train_rmse']:<12.0f} {result['test_rmse']:<11.0f} {overfitting:<12.4f}")
        
        # Track best model
        if result['test_r2'] > best_test_r2:
            best_test_r2 = result['test_r2']
            best_model = model_name
    
    print("-" * 90)
    print(f"🏆 Best performing model: {best_model} (Test R²: {best_test_r2:.4f})")
    
    # Analysis and recommendations
    print(f"\n💡 Analysis:")
    
    linear_gap = linear_results['train_r2'] - linear_results['test_r2']
    ridge_gap = ridge_results['train_r2'] - ridge_results['test_r2']
    lasso_gap = lasso_results['train_r2'] - lasso_results['test_r2']
    elastic_gap = elastic_results['train_r2'] - elastic_results['test_r2']
    
    if linear_gap > 0.1:
        print("  - Linear regression shows overfitting")
    if ridge_gap < linear_gap:
        print("  - Ridge regression reduces overfitting")
    if lasso_gap < linear_gap:
        print("  - Lasso regression reduces overfitting and selects features")
    if elastic_gap < linear_gap:
        print("  - Elastic Net provides balanced regularization")
    
    # Feature selection comparison
    if 'selected_features' in lasso_results:
        total_features = len(lasso_results['model'].coef_)
        print(f"  - Lasso selected {lasso_results['selected_features']}/{total_features} features")
    if 'selected_features' in elastic_results:
        print(f"  - Elastic Net selected {elastic_results['selected_features']}/{total_features} features")

def visualize_regularization_effects(linear_results, ridge_results, lasso_results, 
                                   elastic_results, feature_columns):
    """
    Visualize the effects of different regularization techniques
    """
    print("\n" + "="*70)
    print("STEP 9: Visualizing Regularization Effects")
    print("="*70)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Regularization Effects Comparison', fontsize=16, fontweight='bold')
    
    models = ['Linear', 'Ridge', 'Lasso', 'Elastic Net']
    results = [linear_results, ridge_results, lasso_results, elastic_results]
    colors = ['blue', 'green', 'red', 'orange']
    
    # 1. Performance comparison (R² scores)
    ax1 = axes[0, 0]
    train_scores = [r['train_r2'] for r in results]
    test_scores = [r['test_r2'] for r in results]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, train_scores, width, label='Training R²', alpha=0.7)
    ax1.bar(x + width/2, test_scores, width, label='Testing R²', alpha=0.7)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE comparison
    ax2 = axes[0, 1]
    train_rmse = [r['train_rmse'] for r in results]
    test_rmse = [r['test_rmse'] for r in results]
    
    ax2.bar(x - width/2, train_rmse, width, label='Training RMSE', alpha=0.7)
    ax2.bar(x + width/2, test_rmse, width, label='Testing RMSE', alpha=0.7)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Overfitting comparison
    ax3 = axes[0, 2]
    overfitting_gaps = [r['train_r2'] - r['test_r2'] for r in results]
    
    bars = ax3.bar(models, overfitting_gaps, alpha=0.7, 
                   color=['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' 
                          for gap in overfitting_gaps])
    ax3.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Mild Overfitting')
    ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Significant Overfitting')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Overfitting Gap (Train R² - Test R²)')
    ax3.set_title('Overfitting Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Coefficient comparison for first 3 models
    for i, (model_name, result, color) in enumerate(zip(models[:3], results[:3], colors[:3])):
        ax = axes[1, i]
        coefficients = result['model'].coef_
        
        # Plot coefficients
        feature_indices = range(len(coefficients))
        ax.bar(feature_indices, coefficients, alpha=0.7, color=color)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'{model_name} Coefficients')
        ax.grid(True, alpha=0.3)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 7. Coefficient magnitudes comparison
    ax7 = axes[2, 0]
    for model_name, result, color in zip(models, results, colors):
        coef_magnitudes = np.abs(result['model'].coef_)
        ax7.plot(range(len(coef_magnitudes)), sorted(coef_magnitudes, reverse=True), 
                'o-', label=model_name, color=color, alpha=0.7)
    
    ax7.set_xlabel('Feature Rank')
    ax7.set_ylabel('Coefficient Magnitude')
    ax7.set_title('Coefficient Magnitudes (Sorted)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')
    
    # 8. Feature selection visualization
    ax8 = axes[2, 1]
    feature_counts = []
    for result in results:
        if hasattr(result['model'], 'coef_'):
            selected = np.sum(np.abs(result['model'].coef_) > 1e-10)  # Non-zero coefficients
            feature_counts.append(selected)
        else:
            feature_counts.append(len(feature_columns))
    
    ax8.bar(models, feature_counts, alpha=0.7, color=colors)
    ax8.set_xlabel('Models')
    ax8.set_ylabel('Number of Selected Features')
    ax8.set_title('Feature Selection Comparison')
    ax8.grid(True, alpha=0.3)
    
    # 9. Actual vs Predicted for best model
    ax9 = axes[2, 2]
    best_idx = np.argmax([r['test_r2'] for r in results])
    best_result = results[best_idx]
    best_model_name = models[best_idx]
    
    ax9.scatter(linear_results['y_test_pred'], best_result['y_test_pred'], alpha=0.7)
    
    # Perfect prediction line
    min_val = min(linear_results['y_test_pred'].min(), best_result['y_test_pred'].min())
    max_val = max(linear_results['y_test_pred'].max(), best_result['y_test_pred'].max())
    ax9.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax9.set_xlabel('Linear Regression Predictions')
    ax9.set_ylabel(f'{best_model_name} Predictions')
    ax9.set_title(f'Linear vs {best_model_name} Predictions')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_regularization_paths(X_train_scaled, y_train, ridge_results, lasso_results):
    """
    Analyze how coefficients change with regularization strength
    """
    print("\n" + "="*70)
    print("STEP 10: Analyzing Regularization Paths")
    print("="*70)
    
    # Create regularization path plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Ridge regularization path
    alphas = np.logspace(-4, 2, 100)
    coefs_ridge = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        coefs_ridge.append(ridge.coef_)
    
    coefs_ridge = np.array(coefs_ridge)
    
    for i in range(min(5, coefs_ridge.shape[1])):  # Plot first 5 features
        ax1.plot(alphas, coefs_ridge[:, i], label=f'Feature {i+1}')
    
    ax1.axvline(x=ridge_results['best_alpha'], color='red', linestyle='--', 
                label=f'Best α = {ridge_results["best_alpha"]:.4f}')
    ax1.set_xscale('log')
    ax1.set_xlabel('Alpha (Regularization Strength)')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_title('Ridge Regularization Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lasso regularization path
    coefs_lasso = []
    
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso.fit(X_train_scaled, y_train)
        coefs_lasso.append(lasso.coef_)
    
    coefs_lasso = np.array(coefs_lasso)
    
    for i in range(min(5, coefs_lasso.shape[1])):  # Plot first 5 features
        ax2.plot(alphas, coefs_lasso[:, i], label=f'Feature {i+1}')
    
    ax2.axvline(x=lasso_results['best_alpha'], color='red', linestyle='--', 
                label=f'Best α = {lasso_results["best_alpha"]:.4f}')
    ax2.set_xscale('log')
    ax2.set_xlabel('Alpha (Regularization Strength)')
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title('Lasso Regularization Path')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Regularization Path Analysis:")
    print("  - Ridge: Coefficients shrink smoothly towards zero")
    print("  - Lasso: Coefficients can become exactly zero (feature selection)")
    print(f"  - Best Ridge α: {ridge_results['best_alpha']:.6f}")
    print(f"  - Best Lasso α: {lasso_results['best_alpha']:.6f}")

def main():
    """
    Main function to run the complete regularized regression workflow
    """
    print("🏠 REGULARIZED REGRESSION: RIDGE, LASSO & ELASTIC NET")
    print("🎯 Preventing overfitting and performing feature selection")
    print("="*80)
    
    try:
        # Execute the complete workflow
        data = load_and_prepare_data()
        feature_columns = analyze_multicollinearity(data)
        
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_for_regularization(data, feature_columns)
        
        linear_results = train_baseline_model(X_train_scaled, X_test_scaled, y_train, y_test)
        ridge_results = train_ridge_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        lasso_results = train_lasso_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        elastic_results = train_elastic_net(X_train_scaled, X_test_scaled, y_train, y_test)
        
        compare_all_models(linear_results, ridge_results, lasso_results, elastic_results)
        
        visualize_regularization_effects(linear_results, ridge_results, lasso_results, 
                                       elastic_results, feature_columns)
        
        analyze_regularization_paths(X_train_scaled, y_train, ridge_results, lasso_results)
        
        print("\n" + "="*80)
        print("✅ Regularized Regression analysis completed successfully!")
        print("📊 You learned how regularization prevents overfitting and selects features!")
        print("🎯 Key takeaways:")
        print("   • Ridge (L2): Shrinks coefficients, handles multicollinearity")
        print("   • Lasso (L1): Selects features by setting coefficients to zero")
        print("   • Elastic Net: Combines benefits of both Ridge and Lasso")
        print("="*80)
        
    except FileNotFoundError:
        print("❌ Error: dataset.csv not found!")
        print("Make sure the dataset file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

# Execute the main function when script is run directly
if __name__ == "__main__":
    main()
