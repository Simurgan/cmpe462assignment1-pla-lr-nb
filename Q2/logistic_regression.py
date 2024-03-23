import numpy as np
import matplotlib.pyplot as plt
import random

class LogisticRegression:
    def __init__(self, data_frame, label_frame):
        self.data = data_frame.to_numpy()
        self.labels = label_frame.to_numpy()
        self.weights = np.zeros(self.data.shape[1])
        self.num_iterations = 0

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
        for i in range(self.data.shape[1]):
            min_val = self.data.T[i].min()
            max_val = self.data.T[i].max()
            for d in range(self.data.shape[0]):
                self.data[d][i] = (self.data[d][i] - min_val) / (max_val - min_val)
            
