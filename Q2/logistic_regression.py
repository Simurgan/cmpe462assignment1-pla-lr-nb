import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, data_frame):
        self.data = data_frame.to_numpy()
        np.random.shuffle(self.data)
        
        self.N = self.data.shape[0]
        self.num_features = self.data.shape[1] - 1
        self.reset_weights()

        self.mins = [None] * self.num_features
        self.maxes = [None] * self.num_features

    def plot_feature(self, feature_index):
        data = self.data.T[feature_index]
        min_val = data.min()
        max_val = data.max()
        step = (max_val - min_val) / 100
        heights = [0 for i in range(100)]
        titles = [str(min_val + i * step) for i in range(100)]
        coordinates = [i for i in range(100)]
        for i in range(100):
            for d in data:
                if d >= min_val + i * step and d < min_val + (i + 1) * step:
                    heights[i] += 1

        plt.bar(coordinates, heights, tick_label=titles, width=1)
        plt.xticks(rotation=270)
        plt.show()

    def clip_feature(self, feature_index, min_val, max_val):
        for i in range(len(self.data)):
            if self.data[i][feature_index] < min_val:
                self.data[i][feature_index] = min_val
            elif self.data[i][feature_index] > max_val:
                self.data[i][feature_index] = max_val

    def scale_features(self):
        for i in range(self.num_features):
            min_val = self.data.T[i].min()
            max_val = self.data.T[i].max()
            for d in range(self.N):
                self.data[d][i] = (self.data[d][i] - min_val) / (max_val - min_val)

            self.mins[i] = min_val
            self.maxes[i] = max_val

    def split_data(self, training_ratio):
        training_split = self.data[:int(training_ratio * self.data.shape[0])]
        test_split = self.data[int(training_ratio * self.data.shape[0]):]

        self.N_training = training_split.shape[0]
        self.N_test = test_split.shape[0]

        self.training_data = training_split.T[0:-1].T.astype(np.float64)
        self.test_data = test_split.T[0:-1].T.astype(np.float64)

        self.training_labels = training_split.T[-1]
        self.training_labels[self.training_labels == "Cammeo"] = -1.0
        self.training_labels[self.training_labels == "Osmancik"] = 1.0
        self.training_labels = self.training_labels.reshape((self.N_training, 1)).astype(np.float64)

        self.test_labels = test_split.T[-1]
        self.test_labels[self.test_labels == "Cammeo"] = -1.0
        self.test_labels[self.test_labels == "Osmancik"] = 1.0
        self.test_labels = self.test_labels.reshape((self.N_test, 1)).astype(np.float64)

    def gradient(self, lambda_, idx=None):
        if idx is None:
            exponentials = np.exp(-1 * self.training_labels * (self.training_data @ self.weights))
            factors = -1 * self.training_labels * exponentials / (1 + exponentials)
            gradient = (factors * self.training_data).sum(axis=0) / self.N
        else:
            exponentials = np.exp(-1 * self.training_labels[idx] * (self.training_data[idx] @ self.weights))
            factors = -1 * self.training_labels[idx] * exponentials / (1 + exponentials)
            gradient = factors * self.training_data[idx]

        gradient = gradient.reshape((self.num_features, 1)).astype(np.float64)

        if lambda_ != 0:
            gradient += lambda_ * np.abs(self.weights)
        
        return gradient.reshape((self.num_features, 1)).astype(np.float64)
    
    def reset_weights(self):
        self.weights = np.random.rand(self.num_features, 1).astype(np.float64)

    def evaluate(self):
        training_predictions = np.sign(self.training_data @ self.weights)
        training_accuracy = (training_predictions == self.training_labels).sum() / self.N_training
        
        test_predictions = np.sign(self.test_data @ self.weights)
        test_accuracy = (test_predictions == self.test_labels).sum() / self.N_test

        return training_accuracy, test_accuracy

    def GD(self, step_size, num_iterations, lambda_=0, reset_weights=True):
        if reset_weights:
            self.reset_weights()

        for iteration in range(num_iterations):
            self.weights -= step_size * self.gradient(lambda_)
            training_accuracy, test_accuracy = self.evaluate()

            print("Iteration: ", iteration)
            print("\tTraining accuracy: ", training_accuracy)
            print("\tTest accuracy: ", test_accuracy)

    def SGD(self, step_size, num_epochs, lambda_=0, reset_weights=True):
        if reset_weights:
            self.reset_weights()

        for epoch in range(num_epochs):
            for idx in range(self.N_training):
                self.weights -= step_size * self.gradient(lambda_, idx)
            
            training_accuracy, test_accuracy = self.evaluate()

            print("Epoch: ", epoch)
            print("\tTraining accuracy: ", training_accuracy)
            print("\tTest accuracy: ", test_accuracy)

            
    # Non-regularized
    # E(w) = 1/N * sum(log(1 + exp(-y_n * w^T * x_n)))
    # n: initially specified learning rate
    # n_t: learning rate at the iteration t
    # v^: direction vector
    # v^ = -gradient(E(w)) / ||gradient(E(w))||_2
    # n_t = n * ||gradient(E(w_t))||_2
    # n_t * v^ = -n * gradient(E(w_t))
                
    # Regularized
    # lambda: regularization parameter
    # E(w) = 1/N * sum(log(1 + exp(-y_n * w^T * x_n))) + lambda * w^T * w
                
    # GD Algorithm
    # 1. Initialize w^0
    # 2. For t = 0, 1, 2, ...
    #    a. Compute v^t = gradient(E(w^t))
    #    b. w^{t+1} = w^t - n * v^t
    #    maybe: c. If ||v^t||_2 < epsilon, then stop (copilot suggested)
                
    # SGD Algorithm
    # 1. Initialize w^0
    # 2. pick a random data point from 1 to N
    # 3. run an iteration of GD on that data point
    # w(t+1) = w(t) + y_n * x_n * 1 / (1 + exp(y_n * w^T * x_n))

