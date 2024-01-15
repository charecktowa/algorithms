import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from collections import Counter


class KNN:
    def __init__(self, k: int) -> None:
        self.k = k

    def fit(self, X: list, y: list) -> None:
        self.X = X
        self.y = y

    def predict(self, X: list) -> list:
        return [self._predict(x) for x in X]

    def _predict(self, x: list) -> int:
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1: list, x2: list) -> float:
        return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))


def generate_dataset(samples: int, centers: int, features: int = 2) -> tuple:
    return make_blobs(
        n_samples=samples,
        centers=centers,
        center_box=(0, 8),
        n_features=features,
        random_state=42,
    )


def main():
    X, y = generate_dataset(samples=15, centers=3)

    # Plot X
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap="viridis")
    plt.title("Dataset")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


if __name__ == "__main__":
    main()
