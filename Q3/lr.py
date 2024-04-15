import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, X, y, optimizer="GD", learning_rate=0.01, num_iterations=1000, lambd = 0):
        
        self.X = np.array(X) 
        self.y = np.expand_dims(np.array(y), axis=1)
        self.N = len(self.y)
        self.weights = np.expand_dims(np.random.rand(self.X.shape[1], 1) * 0.01, axis=1)

        self.optimizer = optimizer
        self.step_size = learning_rate
        self.num_iterations = num_iterations
        self.lambd = lambd # regularization parameter
    
        self.num_features = X.shape[1]

    def reset_weights(self):
        self.weights = np.random.rand(self.num_features, 1)*0.01

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        return self._sigmoid(X @ self.weights)
        #return np.sign(self._sigmoid(X @ self.weights))
 
    def gradient(self, idx=None):

        if idx is None:
            exponential = np.exp(-1 * self.y * (self.X @ self.weights)) 
            top = -1 * self.y * exponential
            bottom = 1 + exponential
            gradient = np.sum(top / bottom, axis=0) / self.N
            
            # exponentials = np.exp((-1 * self.y * (self.X @ self.weights))) # (N, 1) (N, 30) @ (30, 1) = (N, 1)
            # factors = -1 * self.y * exponentials / (1 + exponentials)
            # gradient = np.sum(np.dot(factors.T, self.X), axis=0) / self.N
        else:
            exponentials = np.exp(-1 * self.y[idx] * (self.X[idx] @ self.weights))
            factors = -1 * self.y[idx] * exponentials / (1 + exponentials)
            gradient = factors * self.X[idx]

        if self.lambd != 0: # lambd = 0 means no regularization
            regularization_term = np.full_like(self.weights, self.lambd)
            gradient = gradient + self.weights * regularization_term
        
        return gradient

    def GD(self, reset_weights=True):
        if reset_weights:
            self.reset_weights()

        first_time = True
        for i in range(self.num_iterations):
            gradient = self.gradient()
            gradient = np.expand_dims(gradient, axis=1)
            self.weights = self.weights - self.step_size * gradient

    def SGD(self, reset_weights=True):
        if reset_weights:
            self.reset_weights()

        for i in range(self.num_iterations):
            for idx in range(self.N_training):
                self.weights -= self.step_size * self.gradient(idx)

    def train(self):

        if self.optimizer == "GD":
            self.GD()
        elif self.optimizer == "SGD":
            self.SGD()
        else:
            raise ValueError("Invalid optimizer")
        
    def l2_regularization(self):
 
        L2_regularization_cost = (self.lambd/2)*(np.dot(self.weights, self.weights))
        return L2_regularization_cost 
            
