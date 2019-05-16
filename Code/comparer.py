import garcon as gc
import numpy as np
import pandas as pd
import sklearn.svm as svm
import perceptron as pc
import matplotlib.pyplot as plt

DIM = 2

FIG_DIR = '.\\'


class comparer:
    def __init__(self):
        gc.log("Creating comparer")
        self._perc = pc.perceptron()
        self._svm = svm.SVC(C=1e10, kernel='linear')
        self._mu = np.zeros([DIM, 1])
        self._sig = np.eye(DIM)

    def draw_m_points(self, m):
        return np.random.multivariate_normal(mean=self._mu, cov=self._sig,
                                             size=m)

    def compare_one(self, m):
        gc.log("Comparing one")
        fig = plt.figure()
        fig.suptitle("SVM vs Perceptron, " + m + " Samples")
        plt.xlabel('x Coordinate')
        plt.ylabel('y Coordinate')
        points = self.draw_m_points(m)

        plt.legend()
        plt.savefig(FIG_DIR + 'svm_vs_perc_' + m)

    def compare_many(self):
        gc.log("Comparing many")
        for m in [5, 10, 15, 25, 70]:
            self.compare_one(m)
            points = self

    def plot_to_file(self, fn):
        plt.savefig(FIG_DIR + fn)

    def get_true_label(x):
        gc.log("Getting true label of ", x)
        true_w = np.array([0.3, -0.5]).T
        true_b = 0.1
        real_val = np.prod(true_w, x) + true_b
        return np.sign(real_val)


def main():
    pass


main()
