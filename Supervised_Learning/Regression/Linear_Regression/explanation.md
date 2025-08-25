# Linear Regression - Detailed Explanation

## What is Linear Regression?
Linear regression is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data.

## Mathematical Foundation

### Simple Linear Regression (One Feature)
**Equation**: y = mx + b
- y = predicted value (dependent variable)
- x = input feature (independent variable)
- m = slope (coefficient)
- b = y-intercept (bias)

### Multiple Linear Regression (Multiple Features)
**Equation**: y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
- y = predicted value
- x₁, x₂, ..., xₙ = input features
- b₀ = intercept (bias)
- b₁, b₂, ..., bₙ = coefficients for each feature

## Key Assumptions of Linear Regression
1. **Linearity**: The relationship between features and target is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed

## Cost Function
Linear regression uses **Mean Squared Error (MSE)** as the cost function:
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

Where:
- n = number of observations
- yᵢ = actual value
- ŷᵢ = predicted value

## Evaluation Metrics
1. **Mean Squared Error (MSE)**: Average of squared differences
2. **Root Mean Squared Error (RMSE)**: Square root of MSE
3. **Mean Absolute Error (MAE)**: Average of absolute differences
4. **R-squared (R²)**: Proportion of variance explained by the model

## When to Use Linear Regression
- When the relationship between features and target is linear
- For continuous target variables
- When you need an interpretable model
- As a baseline model for comparison

## Advantages
- Simple and fast to train
- No hyperparameters to tune
- Highly interpretable
- Works well with linearly separable data

## Disadvantages
- Assumes linear relationship
- Sensitive to outliers
- Can overfit with too many features
- Requires feature scaling for optimal performance

## Implementation Steps
1. **Data Loading**: Load and explore the dataset
2. **Data Preprocessing**: Handle missing values, scale features
3. **Data Splitting**: Split into training and testing sets
4. **Model Training**: Fit the linear regression model
5. **Prediction**: Make predictions on test data
6. **Evaluation**: Calculate performance metrics
7. **Visualization**: Plot results and residuals
