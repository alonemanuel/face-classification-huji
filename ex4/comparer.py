import os

import garcon as gc
import numpy as np
import pandas as pd
import sklearn.svm as svm
import perceptron as pc
import matplotlib.pyplot as plt

DIM = 2

FIG_DIR = '../Images/'


class Comparer:
    def __init__(self):
        gc.log("Creating comparer")
        self._perc = pc.perceptron()
        self._svm = svm.SVC(C=1e10, kernel='linear')
        self._mu = np.zeros([DIM])
        self._sig = np.eye(DIM)

    def draw_m_points(self, m):
        return np.random.multivariate_normal(mean=self._mu, cov=self._sig,
                                             size=m).T

    def plot_to_file(self, fn):
        plt.savefig(FIG_DIR + fn)

    def true_label(self, x, y):
        true_w = np.array([0.3, -0.5])
        vec = np.array([x, y])
        true_b = 0.1
        real_val = np.matmul(true_w, vec) + true_b
        return np.sign(real_val)

    def draw_svm_hyp(self, X, y):
        classifier = self._svm.fit(X.T, np.ravel(y))
        w = classifier.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-2.5, 2.5)
        yy = a * xx - (classifier.intercept_[0] / w[1])
        plt.plot(xx, yy, label='hyperplane')

    def draw_perc_hyp(self, X, y):
        classifier = self._perc.fit(X.T, np.ravel(y))
        print(classifier)
        w = classifier.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-2.5, 2.5)
        yy = a * xx - (classifier.intercept_[0] / w[1])
        plt.plot(xx, yy, label='hyperplane')

    def compare_one(self, m):
        gc.log("Comparing one")
        fig = plt.figure()
        fig.suptitle("SVM vs Perceptron, " + str(m) + " Samples")
        plt.xlabel('x Coordinate')
        plt.ylabel('y Coordinate')
        points = self.draw_m_points(m)
        raw_labels = np.array([[self.true_label(x, y) for x, y in points.T]])
        labels = np.vstack((points, raw_labels)).T
        good_points = labels[labels[:, 2] == 1]
        bad_points = labels[labels[:, 2] != 1]
        # good_points = labels[bool_labels]
        # bad_points = labels[not bool_labels]
        plt.scatter(good_points[:, 0], good_points[:, 1], label='good',
                    marker='x')
        plt.scatter(bad_points[:, 0], bad_points[:, 1], label='bad', marker='x')
        self.draw_svm_hyp(points, raw_labels)
        self.draw_perc_hyp(points, raw_labels)
        plt.legend()
        plt.savefig(os.path.normpath(FIG_DIR + 'svm_vs_perc_' + str(m)))

    def compare_many(self):
        gc.log("Comparing many")
        for m in [5, 10, 15, 25, 70]:
            self.compare_one(m)
            points = self


def main():
    gc.log("Starting program")
    comp = Comparer()
    comp.compare_many()


main()
