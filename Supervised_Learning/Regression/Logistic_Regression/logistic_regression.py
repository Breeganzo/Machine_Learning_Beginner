"""
Logistic Regression Implementation
=================================

This script demonstrates logistic regression for binary classification.
We'll predict whether a house is expensive (>$350,000) based on its features.

Author: Your Learning Journey
Date: August 25, 2025
"""

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LogisticRegression   # Logistic regression model
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For preprocessing
from sklearn.metrics import (accuracy_score, precision_score, recall_score,  # Classification metrics
                           f1_score, confusion_matrix, roc_auc_score, 
                           roc_curve, classification_report)
import seaborn as sns  # For enhanced visualizations

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """
    Load the dataset and perform initial exploration for classification
    """
    print("="*60)
    print("STEP 1: Loading and Exploring Classification Data")
    print("="*60)
    
    # Load the dataset
    data = pd.read_csv('dataset.csv')
    
    # Display basic information
    print("Dataset shape:", data.shape)
    print("\nColumns:", list(data.columns))
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Analyze the target variable
    print("\nTarget Variable Analysis ('expensive'):")
    target_counts = data['expensive'].value_counts()
    print(target_counts)
    print(f"\nClass distribution:")
    print(f"Not Expensive (0): {target_counts[0]} ({target_counts[0]/len(data)*100:.1f}%)")
    print(f"Expensive (1): {target_counts[1]} ({target_counts[1]/len(data)*100:.1f}%)")
    
    # Check if dataset is balanced
    if abs(target_counts[0] - target_counts[1]) / len(data) < 0.2:
        print("✅ Dataset is relatively balanced")
    else:
        print("⚠️  Dataset is imbalanced - consider balancing techniques")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(data.describe())
    
    return data

def analyze_features_for_classification(data):
    """
    Analyze features specifically for classification task
    """
    print("\n" + "="*60)
    print("STEP 2: Feature Analysis for Classification")
    print("="*60)
    
    # Separate numerical and categorical features
    numerical_features = ['area', 'bedrooms', 'age', 'price']
    categorical_features = ['location']
    
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    
    # Analyze numerical features by class
    print("\n📊 Feature Analysis by Class:")
    for feature in numerical_features:
        if feature != 'price':  # price is used to create target, so skip detailed analysis
            print(f"\n{feature.upper()}:")
            expensive_mean = data[data['expensive'] == 1][feature].mean()
            not_expensive_mean = data[data['expensive'] == 0][feature].mean()
            
            print(f"  Expensive houses (1): {expensive_mean:.2f}")
            print(f"  Not expensive houses (0): {not_expensive_mean:.2f}")
            print(f"  Difference: {expensive_mean - not_expensive_mean:.2f}")
    
    # Analyze categorical features
    print(f"\n📍 LOCATION Distribution:")
    location_cross = pd.crosstab(data['location'], data['expensive'], margins=True)
    print(location_cross)
    
    return numerical_features, categorical_features

def visualize_classification_data(data, numerical_features, categorical_features):
    """
    Create visualizations for classification analysis
    """
    print("\n" + "="*60)
    print("STEP 3: Classification Data Visualization")
    print("="*60)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Logistic Regression: Feature Analysis for Classification', fontsize=16, fontweight='bold')
    
    # Define colors for classes
    colors = ['lightcoral', 'lightblue']
    labels = ['Not Expensive', 'Expensive']
    
    # 1. Target distribution (pie chart)
    target_counts = data['expensive'].value_counts()
    axes[0, 0].pie(target_counts.values, labels=labels, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('Target Variable Distribution')
    
    # 2. Area distribution by class
    for i, class_val in enumerate([0, 1]):
        class_data = data[data['expensive'] == class_val]['area']
        axes[0, 1].hist(class_data, alpha=0.7, label=labels[i], color=colors[i], bins=10)
    axes[0, 1].set_xlabel('House Area (sq ft)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Area Distribution by Class')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Bedrooms distribution by class
    bedroom_data = data.groupby(['bedrooms', 'expensive']).size().unstack(fill_value=0)
    bedroom_data.plot(kind='bar', ax=axes[0, 2], color=colors, alpha=0.7)
    axes[0, 2].set_xlabel('Number of Bedrooms')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Bedrooms Distribution by Class')
    axes[0, 2].legend(labels)
    axes[0, 2].tick_params(axis='x', rotation=0)
    
    # 4. Age distribution by class
    for i, class_val in enumerate([0, 1]):
        class_data = data[data['expensive'] == class_val]['age']
        axes[1, 0].hist(class_data, alpha=0.7, label=labels[i], color=colors[i], bins=10)
    axes[1, 0].set_xlabel('House Age (years)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Age Distribution by Class')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Location distribution by class
    location_cross = pd.crosstab(data['location'], data['expensive'])
    location_cross.plot(kind='bar', ax=axes[1, 1], color=colors, alpha=0.7)
    axes[1, 1].set_xlabel('Location')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Location Distribution by Class')
    axes[1, 1].legend(labels)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Box plot: Area by class
    data.boxplot(column='area', by='expensive', ax=axes[1, 2])
    axes[1, 2].set_xlabel('Expensive (0=No, 1=Yes)')
    axes[1, 2].set_ylabel('Area (sq ft)')
    axes[1, 2].set_title('Area Distribution by Class (Box Plot)')
    
    # 7. Scatter plot: Area vs Price, colored by class
    for i, class_val in enumerate([0, 1]):
        class_data = data[data['expensive'] == class_val]
        axes[2, 0].scatter(class_data['area'], class_data['price'], 
                          alpha=0.7, label=labels[i], color=colors[i])
    axes[2, 0].set_xlabel('Area (sq ft)')
    axes[2, 0].set_ylabel('Price ($)')
    axes[2, 0].set_title('Area vs Price (by Class)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Box plot: Age by class
    data.boxplot(column='age', by='expensive', ax=axes[2, 1])
    axes[2, 1].set_xlabel('Expensive (0=No, 1=Yes)')
    axes[2, 1].set_ylabel('Age (years)')
    axes[2, 1].set_title('Age Distribution by Class (Box Plot)')
    
    # 9. Correlation heatmap
    corr_data = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[2, 2])
    axes[2, 2].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

def prepare_classification_data(data, numerical_features, categorical_features):
    """
    Prepare data for logistic regression
    """
    print("\n" + "="*60)
    print("STEP 4: Data Preparation for Classification")
    print("="*60)
    
    # Create a copy of data for preprocessing
    data_processed = data.copy()
    
    # Handle categorical variables (encode location)
    print("Encoding categorical variables...")
    label_encoder = LabelEncoder()
    data_processed['location_encoded'] = label_encoder.fit_transform(data_processed['location'])
    
    print("Location encoding:")
    location_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    for original, encoded in location_mapping.items():
        print(f"  {original} -> {encoded}")
    
    # Select features for the model (excluding price as it's used to create target)
    feature_columns = ['area', 'bedrooms', 'age', 'location_encoded']
    X = data_processed[feature_columns].values
    y = data_processed['expensive'].values
    
    print(f"\nSelected features: {feature_columns}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Ensure balanced split
    )
    
    print(f"\nData split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Check class distribution in splits
    train_dist = np.bincount(y_train)
    test_dist = np.bincount(y_test)
    print(f"Training class distribution: {train_dist} ({train_dist/len(y_train)*100})")
    print(f"Testing class distribution: {test_dist} ({test_dist/len(y_test)*100})")
    
    # Feature scaling (important for logistic regression)
    print(f"\nApplying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, label_encoder

def train_logistic_regression(X_train_scaled, y_train, feature_columns):
    """
    Train the logistic regression model
    """
    print("\n" + "="*60)
    print("STEP 5: Logistic Regression Training")
    print("="*60)
    
    # Create and train logistic regression model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000  # Increase iterations for convergence
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Extract model parameters
    coefficients = model.coef_[0]  # Coefficients for each feature
    intercept = model.intercept_[0]  # Intercept
    
    print("✅ Model trained successfully!")
    print(f"Intercept: {intercept:.4f}")
    print(f"Coefficients:")
    
    for feature, coef in zip(feature_columns, coefficients):
        print(f"  {feature:15s}: {coef:8.4f}")
    
    # Feature importance analysis
    print(f"\n📊 Feature Importance (coefficient magnitudes):")
    feature_importance = list(zip(feature_columns, np.abs(coefficients)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance, 1):
        print(f"  {i}. {feature:15s}: {importance:.4f}")
    
    # Coefficient interpretation
    print(f"\n🔍 Coefficient Interpretation:")
    for feature, coef in zip(feature_columns, coefficients):
        if coef > 0:
            effect = "increases"
        else:
            effect = "decreases"
        
        # Convert to odds ratio
        odds_ratio = np.exp(coef)
        print(f"  {feature:15s}: {effect} odds by factor of {odds_ratio:.3f}")
    
    return model

def make_predictions_and_probabilities(model, X_test_scaled, y_test):
    """
    Make predictions and get probability estimates
    """
    print("\n" + "="*60)
    print("STEP 6: Making Predictions and Probability Estimates")
    print("="*60)
    
    # Get probability predictions
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
    
    # Get binary predictions (using default threshold of 0.5)
    y_pred = model.predict(X_test_scaled)
    
    print(f"Predictions completed!")
    print(f"Number of predictions: {len(y_pred)}")
    print(f"Probability range: {y_prob.min():.3f} to {y_prob.max():.3f}")
    
    # Show some example predictions
    print(f"\n📋 Sample Predictions:")
    print(f"{'Probability':>12} {'Prediction':>12} {'Actual':>8}")
    print("-" * 35)
    
    for i in range(min(10, len(y_test))):
        print(f"{y_prob[i]:11.3f} {y_pred[i]:11d} {y_test[i]:7d}")
    
    return y_pred, y_prob

def evaluate_classification_model(y_test, y_pred, y_prob):
    """
    Comprehensive evaluation of the logistic regression model
    """
    print("\n" + "="*60)
    print("STEP 7: Model Evaluation")
    print("="*60)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("📊 Model Performance Metrics:")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.1f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Not Exp  Expensive")
    print(f"Actual Not Exp   {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"    Expensive    {cm[1,0]:3d}      {cm[1,1]:3d}")
    
    # Calculate derived metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    print(f"\n🔍 Detailed Analysis:")
    print(f"True Positives (TP):  {tp} - Correctly predicted expensive")
    print(f"True Negatives (TN):  {tn} - Correctly predicted not expensive")
    print(f"False Positives (FP): {fp} - Incorrectly predicted expensive")
    print(f"False Negatives (FN): {fn} - Incorrectly predicted not expensive")
    print(f"Specificity:          {specificity:.4f} - Correctly identified not expensive")
    
    # Performance interpretation
    print(f"\n💡 Performance Interpretation:")
    if accuracy > 0.9:
        print("🌟 Excellent accuracy!")
    elif accuracy > 0.8:
        print("✅ Good accuracy!")
    elif accuracy > 0.7:
        print("👍 Fair accuracy.")
    else:
        print("⚠️  Poor accuracy - model needs improvement.")
    
    if roc_auc > 0.9:
        print("🌟 Excellent discrimination ability!")
    elif roc_auc > 0.8:
        print("✅ Good discrimination ability!")
    elif roc_auc > 0.7:
        print("👍 Fair discrimination ability.")
    else:
        print("⚠️  Poor discrimination - model struggles to separate classes.")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Expensive', 'Expensive']))
    
    return accuracy, precision, recall, f1, roc_auc, cm

def visualize_classification_results(y_test, y_pred, y_prob, model, X_test_scaled, feature_columns):
    """
    Create comprehensive visualizations for classification results
    """
    print("\n" + "="*60)
    print("STEP 8: Results Visualization")
    print("="*60)
    
    # Create comprehensive results figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Logistic Regression Classification Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Not Expensive', 'Expensive'],
                yticklabels=['Not Expensive', 'Expensive'])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Probability Distribution
    prob_expensive = y_prob[y_test == 1]
    prob_not_expensive = y_prob[y_test == 0]
    
    axes[0, 2].hist(prob_not_expensive, alpha=0.7, label='Not Expensive', 
                    color='lightcoral', bins=15, density=True)
    axes[0, 2].hist(prob_expensive, alpha=0.7, label='Expensive', 
                    color='lightblue', bins=15, density=True)
    axes[0, 2].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 2].set_xlabel('Predicted Probability')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Probability Distribution by Class')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Feature Importance (Coefficient Magnitudes)
    coefficients = np.abs(model.coef_[0])
    feature_names = feature_columns
    
    # Sort by importance
    importance_order = np.argsort(coefficients)[::-1]
    sorted_features = [feature_names[i] for i in importance_order]
    sorted_coefficients = coefficients[importance_order]
    
    axes[1, 0].bar(range(len(sorted_features)), sorted_coefficients, 
                   alpha=0.7, color=['blue', 'green', 'red', 'orange'])
    axes[1, 0].set_xlabel('Features')
    axes[1, 0].set_ylabel('Coefficient Magnitude')
    axes[1, 0].set_title('Feature Importance')
    axes[1, 0].set_xticks(range(len(sorted_features)))
    axes[1, 0].set_xticklabels(sorted_features, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Prediction Confidence
    confidence = np.maximum(y_prob, 1 - y_prob)  # Distance from 0.5
    correct_predictions = (y_pred == y_test)
    
    axes[1, 1].scatter(confidence[correct_predictions], [1]*sum(correct_predictions), 
                       alpha=0.6, color='green', label='Correct', s=30)
    axes[1, 1].scatter(confidence[~correct_predictions], [0]*sum(~correct_predictions), 
                       alpha=0.6, color='red', label='Incorrect', s=30)
    axes[1, 1].set_xlabel('Prediction Confidence')
    axes[1, 1].set_ylabel('Prediction Outcome')
    axes[1, 1].set_title('Prediction Confidence vs Accuracy')
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['Incorrect', 'Correct'])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Precision-Recall vs Threshold
    from sklearn.metrics import precision_recall_curve
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_prob)
    
    # Plot precision and recall vs threshold
    axes[1, 2].plot(pr_thresholds, precision_vals[:-1], 'b-', label='Precision', linewidth=2)
    axes[1, 2].plot(pr_thresholds, recall_vals[:-1], 'g-', label='Recall', linewidth=2)
    axes[1, 2].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Default Threshold')
    axes[1, 2].set_xlabel('Threshold')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Precision-Recall vs Threshold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\n🔍 Detailed Prediction Analysis:")
    print("-" * 80)
    print(f"{'Actual':>8} {'Predicted':>10} {'Probability':>12} {'Confidence':>12} {'Correct':>9}")
    print("-" * 80)
    
    for i in range(min(15, len(y_test))):
        actual = y_test[i]
        predicted = y_pred[i]
        probability = y_prob[i]
        confidence = max(probability, 1 - probability)
        is_correct = "✓" if actual == predicted else "✗"
        
        print(f"{actual:7d} {predicted:9d} {probability:11.3f} {confidence:11.3f} {is_correct:>8s}")

def analyze_decision_boundary(model, X_test_scaled, y_test, feature_columns):
    """
    Analyze the decision boundary and feature impact
    """
    print("\n" + "="*60)
    print("STEP 9: Decision Boundary Analysis")
    print("="*60)
    
    # Calculate decision function values
    decision_scores = model.decision_function(X_test_scaled)
    
    print("Decision Function Analysis:")
    print(f"Decision scores range: {decision_scores.min():.3f} to {decision_scores.max():.3f}")
    print(f"Decision boundary at: 0.0")
    
    # Analyze misclassifications
    y_pred = model.predict(X_test_scaled)
    misclassified = y_pred != y_test
    
    if np.any(misclassified):
        print(f"\n❌ Misclassified samples analysis:")
        print(f"Total misclassified: {sum(misclassified)}")
        
        misclassified_indices = np.where(misclassified)[0]
        for idx in misclassified_indices[:5]:  # Show first 5 misclassifications
            actual = y_test[idx]
            predicted = y_pred[idx]
            decision_score = decision_scores[idx]
            features = X_test_scaled[idx]
            
            print(f"\nSample {idx}:")
            print(f"  Actual: {actual}, Predicted: {predicted}")
            print(f"  Decision score: {decision_score:.3f}")
            print(f"  Features: {[f'{val:.2f}' for val in features]}")
    
    # Feature contribution analysis
    print(f"\n🎯 Feature Contribution to Decision:")
    coefficients = model.coef_[0]
    for i, (feature, coef) in enumerate(zip(feature_columns, coefficients)):
        print(f"  {feature:15s}: {coef:8.4f} (weight)")

def main():
    """
    Main function to run the complete logistic regression workflow
    """
    print("🏠 LOGISTIC REGRESSION: HOUSE PRICE CLASSIFICATION")
    print("🎯 Predicting if a house is expensive (>$350,000)")
    print("="*70)
    
    try:
        # Execute the complete workflow
        data = load_and_explore_data()
        numerical_features, categorical_features = analyze_features_for_classification(data)
        visualize_classification_data(data, numerical_features, categorical_features)
        
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, label_encoder = prepare_classification_data(data, numerical_features, categorical_features)
        
        model = train_logistic_regression(X_train_scaled, y_train, feature_columns)
        y_pred, y_prob = make_predictions_and_probabilities(model, X_test_scaled, y_test)
        
        accuracy, precision, recall, f1, roc_auc, cm = evaluate_classification_model(y_test, y_pred, y_prob)
        visualize_classification_results(y_test, y_pred, y_prob, model, X_test_scaled, feature_columns)
        
        analyze_decision_boundary(model, X_test_scaled, y_test, feature_columns)
        
        print("\n" + "="*70)
        print("✅ Logistic Regression Classification completed successfully!")
        print("🎯 Model can predict house price categories with good accuracy!")
        print("="*70)
        
    except FileNotFoundError:
        print("❌ Error: dataset.csv not found!")
        print("Make sure the dataset file is in the same directory as this script.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

# Execute the main function when script is run directly
if __name__ == "__main__":
    main()
