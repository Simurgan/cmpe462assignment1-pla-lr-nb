import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo 
import random
import pandas as pd

# Split dataset into training and testing subsets
def train_test_split(X, test_ratio=0.1, random_seed=None):

    if random_seed:
        random.seed(random_seed)
    
    # Shuffle the indices
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Calculate the number of samples for the testing set
    test_size = int(len(X) * test_ratio)
    train_size = len(X) - test_size
    
    # Create the training and testing subsets
    X_train = X.sample(train_size)
    X_test = X.sample(test_size)
    
    return X_train, X_test

# fetch dataset 
dataset = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = dataset.data.features 
y = dataset.data.targets 

num_classes = len(y.Diagnosis.unique())
num_features = len(X.columns)

set = [X, y]
X = pd.concat(set, axis=1)

X_train, X_test = train_test_split(X, 0.1, 42)

y_train = X_train.Diagnosis
y_train = y_train.to_numpy().reshape(-1, 1)
y_train = np.asarray(y_train, dtype=str)

y_test = X_test.Diagnosis
y_test = y_test.to_numpy().reshape(-1, 1)
y_test= np.asarray(y_test, dtype=str)

class GaussianNaiveBayes:
    def __init__(self):
        self.class_prior = None
        self.class_mean = None
        self.class_std = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_prior = np.zeros(n_classes)
        self.class_mean = np.zeros((n_classes, n_features))
        self.class_std = np.zeros((n_classes, n_features))

        # Calculate class prior probabilities
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_prior[i] = len(X_c) / n_samples

            # Calculate mean and standard deviation for each class and feature
            self.class_mean[i, :] = X_c.mean(axis=0)
            self.class_std[i, :] = X_c.std(axis=0)

    def _gaussian_pdf(self, X, mean, std):
        exponent = np.exp(-((X - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def _predict_sample(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for i, c in enumerate(self.classes):
            prior = np.log(self.class_prior[i])
            likelihood = np.sum(np.log(self._gaussian_pdf(x, self.class_mean[i], self.class_std[i])))
            posterior = prior + likelihood
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        # Predict the class for each sample
        return np.array([self._predict_sample(x) for x in X])


# Train Full Bayes classifier
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

# Predict
y_pred = gnb.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

