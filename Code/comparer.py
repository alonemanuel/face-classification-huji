import garcon as gc
import numpy as np
import pandas as pd
import sklearn.svm as svm
import perceptron as pc
import matplotlib.pyplot as plt

DIM = 2

FIG_DIR = '..\\Images\\'


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

    def compare_one(self, m):
        gc.log("Comparing one")
        fig = plt.figure()
        fig.suptitle("SVM vs Perceptron, " + str(m) + " Samples")
        plt.xlabel('x Coordinate')
        plt.ylabel('y Coordinate')
        x, y = self.draw_m_points(m)
        stacked = np.vstack((x, y)).T
        gc.log('Stacked shape is ', stacked.shape)
        labels = [self.true_label(x, y) for x,y in stacked]
        plt.scatter(x, y, c=labels)
        # plt.legend()
        plt.savefig(FIG_DIR + 'svm_vs_perc_' + str(m))

    def compare_many(self):
        gc.log("Comparing many")
        for m in [5, 10, 15, 25, 70]:
            self.compare_one(m)
            points = self

    def plot_to_file(self, fn):
        plt.savefig(FIG_DIR + fn)

    def true_label(self, x, y):
        gc.log("Getting true label of ")
        true_w = np.array([[0.3, -0.5]]).T
        true_b = 0.1
        gc.log("w's shape is ", true_w.shape)
        gc.log("x's shape is ", x.shape)
        real_val = np.prod(true_w, np.array([x, y])) + true_b
        return np.sign(real_val)


def main():
    comp = Comparer()
    comp.compare_many()


main()
