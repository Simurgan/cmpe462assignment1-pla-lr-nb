import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate synthetic dataset
X, y = make_classification(n_samples=500, n_features=30, n_classes=2, random_state=42)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define logistic regression function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, theta):
    z = np.dot(X, theta)
    return sigmoid(z)

# Define function for calculating accuracy
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Define k-fold cross-validation function
def k_fold_cross_validation(X, y, k, C_values):
    n_samples = X.shape[0]
    fold_size = n_samples // k

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    # Initialize arrays to store scores for each value of C
    mean_scores = np.zeros(len(C_values))

    for i, C in enumerate(C_values):
        scores = []

        for j in range(k):
            # Split data into training and validation folds
            val_indices = indices[j * fold_size: (j + 1) * fold_size]
            train_indices = np.concatenate([indices[:j * fold_size], indices[(j + 1) * fold_size:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            # Train logistic regression model
            theta = np.zeros(X_train.shape[1])
            for _ in range(100):  # 100 iterations for simplicity
                predictions = logistic_regression(X_train, y_train, theta)
                gradient = np.dot(X_train.T, (predictions - y_train)) / len(y_train)
                theta -= C * gradient

            # Predict on validation fold
            val_predictions = (logistic_regression(X_val, y_val, theta) >= 0.5).astype(int)

            # Calculate accuracy
            score = accuracy_score(y_val, val_predictions)
            scores.append(score)

        # Calculate mean score for current value of C
        mean_scores[i] = np.mean(scores)

    # Select optimal value of C
    optimal_C_index = np.argmax(mean_scores)
    optimal_C = C_values[optimal_C_index]

    return optimal_C

# Define candidate values for regularization parameter C
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Perform k-fold cross-validation
optimal_C = k_fold_cross_validation(X, y, k=5, C_values=C_values)

print("Optimal Regularization Parameter (C):", optimal_C)
