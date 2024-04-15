import numpy as np

class NB:
    def __init__(self, X, y):
        self.num_examples = X.shape[0]
        self.num_features = 30
        self.num_classes = 2
        self.classes = ["M", "B"] # np.unique(y)
        self.eps = 1e-6

    def fit(self, X):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in self.classes: # classes are 0, 1, 2, etc
            X_c = X[X["Diagnosis"] == c] # all examples that belong to class c

            self.classes_mean[str(c)] = np.mean(X_c, axis=0) # mean of each feature for class c
            self.classes_variance[str(c)] = np.var(X_c, axis=0) # variance of each feature for class c
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0] # prior probability of class c    num of examples in class c / total num of examples

    def predict(self, X):
        probs = np.zeros((self.num_examples, self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.density_function(X, self.classes_mean[str(c)], self.classes_variance[str(c)])
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)

    def density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.eps))
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs
    
    def log_density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        val = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * self.num_features * np.log(np.power(sigma + self.eps, 2)) - 0.5 * np.sum(np.power(x - mean, 2) / (np.power(sigma + self.eps, 2)), 1)
        return val



# if __name__ == "__main__":
#     X = np.loadtxt("example_data/data.txt", delimiter=",")
#     y = np.loadtxt("example_data/targets.txt") - 1
# 
#     NB = NaiveBayes(X, y)
#     NB.fit(X)
#     y_pred = NB.predict(X)
# 
#     print(f"Accuracy: {sum(y_pred==y)/X.shape[0]}")