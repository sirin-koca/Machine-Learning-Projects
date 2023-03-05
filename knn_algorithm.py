import numpy as np

"""The Euclidean distance is a measure of the distance between two points in a multidimensional space, 
and is commonly used in machine learning to compute the similarity between two data points.

For example, if x1 = [1, 2, 3] and x2 = [4, 5, 6], the Euclidean distance between them would be:
d(x1, x2) = sqrt((1 - 4)^2 + (2 - 5)^2 + (3 - 6)^2) = sqrt(27) = 5.196
"""


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.y_train = None
        self.X_train = None
        self.k = k

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def predict(self, x):
        y_pred = []
        for x in x:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            y_pred.append(most_common)
        return y_pred
