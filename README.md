# CSE 3683 Lab 3: Machine Learning for Housing Price Prediction

A comprehensive machine learning lab implementing housing price prediction using the California Housing dataset. This lab explores both regression and classification techniques through Linear Regression and K-Nearest Neighbors algorithms.

## 📋 Lab Overview

This lab is divided into three main parts:
- **Part A**: Data Loading and Visualization
- **Part B**: Linear Regression (Simple and Multiple)
- **Part C**: Classification using K-Nearest Neighbors

## 🏠 Dataset: California Housing Prices

The lab uses the California housing dataset from the 1990 U.S. census, available through scikit-learn.

### Dataset Characteristics:
- **Number of Instances**: 20,640
- **Number of Features**: 8
- **Target Variable**: Median house value (in $100,000s)
- **Data Split**: 80% training, 20% testing

### Features:
1. **MedInc**: Median income in block group
2. **HouseAge**: Median house age in block group
3. **AveRooms**: Average number of rooms per household
4. **AveBedrms**: Average number of bedrooms per household
5. **Population**: Block group population
6. **AveOccup**: Average number of household members
7. **Latitude**: Block group latitude
8. **Longitude**: Block group longitude

## 🚀 Part A: Data Loading and Visualization

### Data Loading
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load dataset
dataset = fetch_california_housing()
features = dataset.data
labels = dataset.target

# Split data (80% train, 20% test)
train_x, test_x, train_y, test_y = train_test_split(
    features, labels, test_size=0.2, random_state=0
)
```

### Data Exploration
- **Feature distribution analysis**: 2×4 grid of histograms showing distribution of all 8 features
- **Target variable analysis**: Histogram of median house prices
- **Data shape verification**: Confirm training and testing set dimensions

### Visualization Components:
- Feature histograms to understand data distribution
- Target variable histogram to understand price ranges
- Data shape confirmation for proper splitting

## 📈 Part B: Linear Regression

### Simple Linear Regression

Implementation of simple linear regression from scratch using the median income feature.

#### Mathematical Foundation:
```
θ₁ = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
θ₀ = ȳ - θ₁x̄
```

#### Implementation:
```python
def simple_linear_regression_fit(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate covariance and variance
    cov = np.sum((x - x_mean) * (y - y_mean))
    var = np.sum((x - x_mean) ** 2)
    
    # Calculate slope and intercept
    theta1 = cov / var  # slope
    theta0 = y_mean - theta1 * x_mean  # intercept
    
    return theta0, theta1
```

#### Features:
- **Single feature analysis**: Uses median income as predictor
- **Visual comparison**: Side-by-side plots of training and test predictions
- **Manual implementation**: Built from mathematical foundations

### Multiple Linear Regression

Using scikit-learn's LinearRegression for all 8 features.

```python
from sklearn.linear_model import LinearRegression

# Create and train model
model = LinearRegression()
model.fit(train_x, train_y)

# Make predictions
predicted_y_test = model.predict(test_x)
```

#### Features:
- **All features utilized**: Incorporates all 8 housing characteristics
- **Scikit-learn implementation**: Leverages optimized library functions
- **Prediction visualization**: Scatter plot comparing predicted vs. actual prices

## 🎯 Part C: Classification with K-Nearest Neighbors

### Problem Transformation
The regression problem is converted to binary classification by creating a threshold-based class system.

### K-NN Implementation

Custom implementation of the K-Nearest Neighbors algorithm:

```python
def k_nearest_neighbor(train_x, train_y_class, test_x, K=3):
    predicted_y_test = np.zeros(len(test_x), dtype=bool)
    
    for i in range(len(test_x)):
        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((train_x - test_x[i])**2, axis=1))
        
        # Find K nearest neighbors
        nearest_indices = np.argpartition(distances, K)[:K]
        nearest_labels = train_y_class[nearest_indices]
        
        # Majority voting
        predicted_y_test[i] = np.mean(nearest_labels) > 0.5
    
    return predicted_y_test
```

#### Algorithm Features:
- **Distance-based classification**: Uses Euclidean distance metric
- **Majority voting**: Classifies based on majority class of K neighbors
- **Configurable K**: Allows tuning of neighborhood size
- **Custom implementation**: Built from algorithmic foundations

### Accuracy Evaluation

```python
def get_accuracy(predicted_y, true_y):
    correct = np.sum(predicted_y == true_y)
    total = len(true_y)
    return correct / total
```

## 🛠️ Prerequisites

### Required Libraries:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

### Installation:
```bash
pip install numpy matplotlib scikit-learn
```

## 📊 Expected Results

### Linear Regression Performance:
- **Simple Linear Regression**: Shows relationship between median income and house prices
- **Multiple Linear Regression**: Improved predictions using all features
- **Visualization**: Scatter plots showing prediction accuracy

### K-NN Classification Performance:
- **Binary classification**: High/low price category prediction
- **Accuracy metrics**: Percentage of correct classifications
- **Tunable parameters**: K value optimization

## 🔧 Key Implementation Details

### Data Preprocessing:
- **Train-test split**: 80-20 ratio with fixed random state
- **Feature scaling**: May be needed for K-NN (distance-based)
- **Class creation**: Converting continuous prices to binary categories

### Algorithm Implementations:
- **Manual linear regression**: Understanding mathematical foundations
- **Library-based regression**: Leveraging optimized implementations
- **Custom K-NN**: Distance calculation and voting mechanisms

### Visualization Techniques:
- **Histogram analysis**: Feature and target distributions
- **Scatter plots**: Prediction vs. actual comparisons
- **Side-by-side plots**: Training vs. testing performance

## 📚 Learning Objectives

### Regression Concepts:
- **Simple vs. Multiple Linear Regression**: Understanding complexity trade-offs
- **Coefficient interpretation**: Meaning of slopes and intercepts
- **Prediction visualization**: Assessing model performance graphically

### Classification Concepts:
- **Distance-based learning**: K-NN algorithm principles
- **Majority voting**: Democratic decision-making in ML
- **Binary classification**: Converting regression to classification problems

### Machine Learning Fundamentals:
- **Train-test splits**: Proper evaluation methodology
- **Feature analysis**: Understanding data characteristics
- **Performance metrics**: Accuracy calculation and interpretation

## 🎯 Extensions and Improvements

### Advanced Techniques:
```python
# Feature scaling for K-NN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# Cross-validation for K selection
from sklearn.model_selection import cross_val_score
# Test different K values

# Regularized regression
from sklearn.linear_model import Ridge, Lasso
ridge_model = Ridge(alpha=1.0)
lasso_model = Lasso(alpha=1.0)
```

### Potential Enhancements:
1. **Feature Engineering**: Create new features from existing ones
2. **Hyperparameter Tuning**: Optimize K value for K-NN
3. **Cross-Validation**: More robust model evaluation
4. **Feature Scaling**: Improve K-NN performance
5. **Regularization**: Prevent overfitting in linear regression

## 📈 Performance Analysis

### Evaluation Metrics:
- **Regression**: Mean Squared Error, R-squared
- **Classification**: Accuracy, Precision, Recall
- **Visual Assessment**: Scatter plots, residual plots

### Comparison Framework:
- **Simple vs. Multiple Regression**: Feature importance analysis
- **Different K values**: Bias-variance trade-off in K-NN
- **Algorithm comparison**: Regression vs. classification approaches

## 🔍 Troubleshooting

### Common Issues:
1. **Data shape mismatches**: Verify train-test split dimensions
2. **Import errors**: Ensure all required libraries are installed
3. **Visualization issues**: Check matplotlib backend configuration
4. **Accuracy calculation**: Verify boolean array operations

### Debugging Tips:
- Print intermediate shapes and values
- Visualize data distributions before modeling
- Check for NaN or infinite values
- Verify random state consistency

---

This lab provides hands-on experience with fundamental machine learning algorithms, demonstrating both the mathematical foundations and practical implementation of regression and classification techniques on real-world housing data.
