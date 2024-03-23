import numpy as np
import matplotlib.pyplot as plt
import random

class PLA:
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)
        self.weights = np.zeros(3)
        self.num_iterations = 0

    def scatter_data(self):
        positiveX = []
        positiveY = []
        negativeX = []
        negativeY = []
        for idx, label in enumerate(self.labels):
            if label == 1:
                positiveX.append(self.data[idx][1])
                positiveY.append(self.data[idx][2])
            elif label == -1:
                negativeX.append(self.data[idx][1])
                negativeY.append(self.data[idx][2])
                
        plt.scatter(positiveX, positiveY, c='b', marker='o')
        plt.scatter(negativeX, negativeY, c='r', marker='x')
        plt.show()
        
    def get_random_misclassified_sample(self):
        misclassifiedSamples = []
        for idx, x in enumerate(self.data):
            if np.sign(np.dot(self.weights, x)) != self.labels[idx]:
                misclassifiedSamples.append(self.data[idx] * self.labels[idx])
        if len(misclassifiedSamples) == 0:
            return None
        else:
            return random.choice(misclassifiedSamples)
            
    def train(self, initial_weights=None):
        if initial_weights is not None:
            self.weights = np.array(initial_weights)
        self.num_iterations = 0
        misclassifiedSample = self.get_random_misclassified_sample()
        while misclassifiedSample is not None:
            self.weights += misclassifiedSample
            misclassifiedSample = self.get_random_misclassified_sample()
            self.num_iterations += 1
            
    def plot_decision_boundary(self):
        positiveX = []
        positiveY = []
        negativeX = []
        negativeY = []
        for idx, label in enumerate(self.labels):
            if label == 1:
                positiveX.append(self.data[idx][1])
                positiveY.append(self.data[idx][2])
            elif label == -1:
                negativeX.append(self.data[idx][1])
                negativeY.append(self.data[idx][2])
                
        plt.scatter(positiveX, positiveY, c='b', marker='o')
        plt.scatter(negativeX, negativeY, c='r', marker='x')
        plt.plot([0, -self.weights[0]/self.weights[1]], [-self.weights[0]/self.weights[2], 0], 'k-')
        plt.show()

    def print_last_training_stats(self):
        print(f'Number of iterations: {self.num_iterations}')
        print(f'Final weights: {self.weights}')