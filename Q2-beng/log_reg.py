import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, X, y, optimizer="GD", learning_rate=0.01, num_iterations=1000, lambd = 0):

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.step_size = learning_rate
        self.num_iterations = num_iterations

        self.lambd = lambd # regularization parameter
        self.X = X
        self.y = y
        self.N = self.data.shape[0]
        self.num_features = self.data.shape[1] - 1
        self.weights = np.random.rand(self.num_features, 1) * 0.01

    def reset_weights(self):
        self.weights = np.random.rand(self.num_features, 1)*0.01

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        return self._sigmoid(X @ self.weights)
 
    def gradient(self, idx=None):

        if idx is None:
            exponentials = np.exp(-1 * self.y * (self.X @ self.weights))
            factors = -1 * self.y * exponentials / (1 + exponentials)
            gradient = (factors * self.X).sum(axis=0) / self.N
        else:
            exponentials = np.exp(-1 * self.y[idx] * (self.X[idx] @ self.weights))
            factors = -1 * self.y[idx] * exponentials / (1 + exponentials)
            gradient = factors * self.X[idx]

        if self.lambd != 0: # lambd = 0 means no regularization
            gradient += self.lambd * self.weights
        
        return gradient 

    def GD(self, reset_weights=True):
        if reset_weights:
            self.reset_weights()

        for i in range(self.num_iterations):
            self.weights -= self.step_size * self.gradient()

    def SGD(self, reset_weights=True):
        if reset_weights:
            self.reset_weights()

        for i in range(self.num_iterations):
            for idx in range(self.N_training):
                self.weights -= self.step_size * self.gradient(idx)

    def _update(self):

        if self.optimizer == "GD":
            self.GD()
        elif self.optimizer == "SGD":
            self.SGD()
        else:
            raise ValueError("Invalid optimizer")
        
    def l2_regularization(self):
 
        L2_regularization_cost = (self.lambd/2)*(np.dot(self.weights, self.weights))
        return L2_regularization_cost 
            
