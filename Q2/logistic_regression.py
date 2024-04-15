import math
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

    #def split_data_to_k_groups(self, training_ratio, k):

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

    def gradient(self, lambda_power, idx=None, training_data=None, training_labels=None):
        if training_data is None:
            training_data = self.training_data

        if training_labels is None:
            training_labels = self.training_labels

        if idx is None:
            exponentials = np.exp(-1 * training_labels * (training_data @ self.weights))
            factors = -1 * training_labels * exponentials / (1 + exponentials)
            gradient = (factors * training_data).sum(axis=0) / len(training_labels)
        else:
            exponentials = np.exp(-1 * training_labels[idx] * (training_data[idx] @ self.weights))
            factors = -1 * training_labels[idx] * exponentials / (1 + exponentials)
            gradient = factors * training_data[idx]

        gradient = gradient.reshape((self.num_features, 1)).astype(np.float64)

        if lambda_power is not None:
            gradient += ((10**lambda_power) * np.abs(self.weights))
        
        return gradient.reshape((self.num_features, 1)).astype(np.float64)
    
    def reset_weights(self):
        self.weights = (0.01 * np.random.randn(self.num_features, 1)).astype(np.float64)

    def evaluate(self, training_data=None, training_labels=None, test_data=None, test_labels=None):
        if training_data is None:
            training_data = self.training_data
        if training_labels is None:
            training_labels = self.training_labels
        if test_data is None:
            test_data = self.test_data
        if test_labels is None:
            test_labels = self.test_labels

        training_predictions = np.sign(training_data @ self.weights)
        training_accuracy = (training_predictions == training_labels).sum() / len(training_labels)
        
        test_predictions = np.sign(test_data @ self.weights)
        test_accuracy = (test_predictions == test_labels).sum() / len(test_labels)

        return training_accuracy, test_accuracy
    
    def calculate_loss(self, lambda_power=None):
        exponentials = np.exp(-1 * self.training_labels * (self.training_data @ self.weights))
        loss = np.log(1 + exponentials).sum(axis=0) / self.N_training
        loss = loss.reshape((1, 1)).astype(np.float64)
        if lambda_power is not None:
            loss += ((10**lambda_power) / 2) * (self.weights.T @ self.weights)

        return loss[0][0]

    def GD(self, step_size, num_iterations, lambda_power=None, reset_weights=True):
        if reset_weights:
            self.reset_weights()
            self.losses = []

        for iteration in range(num_iterations):
            self.weights -= step_size * self.gradient(lambda_power)
            training_accuracy, test_accuracy = self.evaluate()

            loss = self.calculate_loss(lambda_power=lambda_power)
            self.losses.append(loss)

            print("Iteration: ", iteration)
            print("\tLoss: ", loss)
            print("\tTraining accuracy: ", training_accuracy)
            print("\tTest accuracy: ", test_accuracy)

    def SGD(self, step_size, num_epochs, lambda_power=None, reset_weights=True):
        if reset_weights:
            self.reset_weights()
            self.losses = []

        for epoch in range(num_epochs):
            for idx in range(self.N_training):
                self.weights -= step_size * self.gradient(lambda_power, idx)
            
            training_accuracy, test_accuracy = self.evaluate()

            loss = self.calculate_loss(lambda_power=lambda_power)
            self.losses.append(loss)

            print("Epoch: ", epoch)
            print("\tLoss: ", loss)
            print("\tTraining accuracy: ", training_accuracy)
            print("\tTest accuracy: ", test_accuracy)

    def find_best_lambda(self, fold_num, lambda_powers, step_size, num_epochs, is_stochastic=False):
        best_lambda = None
        best_accuracy = 0
        mean_accuracies = []
        
        data_folds = np.array_split(self.training_data, fold_num)
        label_folds = np.array_split(self.training_labels, fold_num)
        
        for lambda_power in lambda_powers:
            accuracies = []

            for j in range(fold_num):
                valid_data = data_folds[j]
                valid_labels = label_folds[j]

                train_data = np.concatenate([data_folds[k] for k in range(fold_num) if k != j])
                train_labels = np.concatenate([label_folds[k] for k in range(fold_num) if k != j])

                self.reset_weights()
                if not is_stochastic:
                    for iteration in range(num_epochs):
                        self.weights -= step_size * self.gradient(lambda_power, training_data=train_data, training_labels=train_labels)
                        training_accuracy, test_accuracy = self.evaluate(training_data=train_data, training_labels=train_labels, test_data=valid_data, test_labels=valid_labels)
                else:
                    for epoch in range(num_epochs):
                        for idx in range(len(train_labels)):
                            self.weights -= step_size * self.gradient(lambda_power, idx, training_data=train_data, training_labels=train_labels)
                    
                    training_accuracy, test_accuracy = self.evaluate(training_data=train_data, training_labels=train_labels, test_data=valid_data, test_labels=valid_labels)
                
                accuracies.append(test_accuracy)
            
            mean_accuracy = np.mean(accuracies)
            mean_accuracies.append(mean_accuracy)

            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_lambda = lambda_power
        
        return best_lambda, mean_accuracies
