# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:12:46 2017

@author: alex
"""

import numpy as np
import sklearn.datasets
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from time import time


def vis_data(data, labels):
    colors_d = {0: 'g', -1: 'g', 1: 'b'}
    colors = [colors_d[x] for x in labels]
    for label in set(labels):
        x1, x2 = data[labels == label, 0], data[labels == label, 1]
        plt.scatter(x1, x2,
                    marker='o',
                    s=100,
                    edgecolor=colors_d[label],
                    c='none',
                    alpha=0.7,
                    linewidths=2,
                    label="Class" + str(label))
    plt.legend()
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()


class SimpleSvm:
    x = None
    y = None
    n = None
    _kernel_type = ''
    alphas = None
    b = None
    C = None
    support_vectors = None
    tolerance = None
    max_passes = None
    _time_to_fit = None
    _random_seed = None

    def __init__(self, C=1.0, tolerance=0.05, max_passes=20, kernel='linear', random_seed=1, verbose=0):
        self.C = C
        self.tolerance = tolerance
        self.max_passes = max_passes
        self._kernel_type = kernel
        self._random_seed = random_seed
        self.verbose = verbose

    def fit(self, x, y):
        self._verify_data(x, y)

        st = time()
        self.x = x
        self.y = y
        self.n = len(y)
        if self._random_seed:
            np.random.seed(self._random_seed)

        self.alphas = np.zeros(self.n)
        self.b = 0
        passes_count = 0

        while passes_count < self.max_passes:
            num_changed_alphas = 0
            # for each instance
            for i in range(self.n):
                E_i = self._calc_f_x(self.x[i]) - self.y[i]
                if (self.y[i] * E_i < -self.tolerance and self.alphas[i] < self.C) or (
                            self.y[i] * E_i > self.tolerance and self.alphas[i] > 0):
                    rand_ind = np.random.randint(low=0, high=10, size=2)
                    j = rand_ind[0] if rand_ind[0] != i else rand_ind[1]

                    E_j = self._calc_f_x(self.x[j]) - self.y[j]
                    a_i, a_j = self.alphas[i], self.alphas[j]

                    # L and H
                    if self.y[i] != self.y[j]:
                        L = max(0.0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0.0, self.alphas[j] + self.alphas[i] - self.C)
                        H = min(self.C, self.alphas[j] + self.alphas[i])

                    if L == H:
                        continue

                    # small eta
                    h = 2 * self._kernel(self.x[i], self.x[j]) - \
                        self._kernel(self.x[i], self.x[i]) - \
                        self._kernel(self.x[j], self.x[j])
                    if h >= 0:
                        continue

                    # new a_j
                    self.alphas[j] = self.alphas[j] - (self.y[j] * (E_i - E_j)) / h
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif L <= self.alphas[j] <= H:
                        self.alphas[j] = self.alphas[j]
                    elif self.alphas[j] < L:
                        self.alphas[j] = L

                    if abs(a_j - self.alphas[j]) < 10 ** -5:
                        continue

                    # new a_i
                    self.alphas[i] = self.alphas[i] + self.y[i] * self.y[j] * (a_j - self.alphas[j])

                    # new b
                    b1 = self.b - E_i - self.y[i] * (self.alphas[i] - a_i) * self._kernel(self.x[i], self.x[i]) - \
                         self.y[j] * (self.alphas[j] - a_j) * self._kernel(self.x[i], self.x[j])
                    b2 = self.b - E_j - self.y[i] * (self.alphas[i] - a_i) * self._kernel(self.x[i], self.x[j]) - \
                         self.y[j] * (self.alphas[j] - a_j) * self._kernel(self.x[j], self.x[j])

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed_alphas += 1
                    if self.verbose > 0:
                        print("num_changed_alphas = %d" % num_changed_alphas)
            if num_changed_alphas == 0:
                passes_count += 1
            else:
                passes_count = 0

        self._time_to_fit = time() - st
        # save support vectors ids
        self.support_vectors = np.where(self.alphas > 0)[0]

    def predict(self, x):
        fx = [self._calc_f_x(xi) for xi in x]
        predictions = [-1 if fxi < 0 else 1 for fxi in fx]
        return predictions

    def get_tr_accuracy(self):
        predictions = self.predict(self.x)
        return metrics.accuracy_score(self.y, predictions)

    def compare_to_sklearn(self):
        sklearn_model = sklearn.svm.SVC(C=self.C, kernel=self._kernel_type)
        st = time()
        sklearn_model.fit(X=self.x, y=self.y)
        time_sklearn = time() - st
        sklearn_prediction = sklearn_model.predict(X=self.x)
        sklearn_prediction[sklearn_prediction == 0] = -1  # although sklearn.svm also use -1/1 convention

        my_prediction = self.predict(x=self.x)

        # compare #
        mult_pred = my_prediction * sklearn_prediction
        n_right = sum(mult_pred > 0)
        print("\nSimpleSvm vs sklearn.svm.SVC:\n=============================\n")
        print("n={} | equal={} | unequal={}".format(self.n, n_right, self.n-n_right))
        print("Train accuracy:      SimpleSvm={} | sklearn={}".
              format(self.get_tr_accuracy(), sklearn_model.score(self.x, self.y)))
        print("Num support vectors: SimpleSvm={} | sklearn={}".
              format(len(self.support_vectors), len(sklearn_model.support_vectors_)))
        print("Training time:       SimpleSvm={0:.3f} sec | sklearn={1:.3f} sec".format(self._time_to_fit, time_sklearn))
        return sklearn_model

    def visualize_model(self):
        colors_d = {0: 'g', -1: 'g', 1: 'b'}
        colors = [colors_d[val] for val in self.y]
        plt.scatter(self.x[:, 0], self.x[:, 1],
                    marker='o',
                    s=80,
                    edgecolors=colors,
                    c='none',
                    alpha=0.7,
                    linewidths=2)
        a, b = self._get_sv_by_class()
        plt.plot(a[:, 0], a[:, 1], marker='d', linestyle='', color='g', label='support vectors -1')
        plt.plot(b[:, 0], b[:, 1], marker='d', linestyle='', color='b', label='support vectors +1')
        plt.legend()
        plt.show()

    def _calc_f_x(self, x_val):
        kernels_v = np.array([self._kernel(x_val, self.x[x]) if self.alphas[x] else 0 for x in range(self.n)])
        f_x = sum(self.alphas * self.y * kernels_v) + self.b
        return f_x

    def _kernel(self, a, b):
        if self._kernel_type == 'linear':
            return sum(a * b)
        else:
            raise Exception("Unsupported kernel")
            return None

    def _get_sv_by_class(self):
        sv_x = [self.x[xi] for xi in self.support_vectors]
        sv_classes = [self.y[xi] for xi in self.support_vectors]  # self.predict(sv_x)
        a = np.array([sv_x[i] for i, pred in enumerate(sv_classes) if pred == -1])
        b = np.array([sv_x[i] for i, pred in enumerate(sv_classes) if pred == 1])
        return a, b

    def _verify_data(self, x, y):
        if len(x) != len(y):
            print("X and Y contain different number of samples!")
            return False
        if len(np.unique(y)) > 2:
            print("Currently only 2 classes support!")
            return False
        if 0 in y:
            y[y == 0] = -1
            return True


def main(cluster_std=2.5):
    # Generate random dataset with 2 classes in 2D space
    data, labels = sklearn.datasets.make_blobs(centers=2, cluster_std=cluster_std)

    # Show the data
    vis_data(data, labels)

    # Fit simple SVM model
    my_svm = SimpleSvm()
    my_svm.fit(data, labels)

    # Compare to sklearn.svm.SVC
    sk_svm = my_svm.compare_to_sklearn()

if "__name__" == "__main__":
    main()
main()
