import os

import garcon as gc
import numpy as np
import pandas as pd
import sklearn.svm as svm
import perceptron as pc
import matplotlib.pyplot as plt

DIM = 2

FIG_DIR1 = '../Images1/'
FIG_DIR2 = '../Images2/'


class Comparer:
    def __init__(self):
        gc.log("Creating comparer")
        self._perc = pc.Perceptron()
        self._svm = svm.SVC(C=1e10, kernel='linear')
        self._mu = np.zeros([DIM])
        self._sig = np.eye(DIM)

    def draw_m_points(self, m):
        # gc.log("Drawing ", m, " points")
        return np.random.multivariate_normal(mean=self._mu, cov=self._sig,
                                             size=m).T

    def plot_to_file(self, fn, dirnum=1):
        plt.savefig((FIG_DIR1 if dirnum == 1 else FIG_DIR2) + fn)

    def true_label(self, X):
        true_w = np.array([[0.3], [-0.5]])
        true_b = 0.1
        real_val = np.matmul(X, true_w) + true_b
        return np.sign(real_val)

    def draw_svm_hyp(self, X, y):
        gc.log("Drawing SVM")
        classifier = self._svm.fit(X.T, np.ravel(y))
        w = classifier.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-2.5, 2.5)
        yy = a * xx - (classifier.intercept_[0] / w[1])
        plt.plot(xx, yy, label='SVM', color='red')

    def get_svm_accu(self, svm, X, y):
        w = svm.coef_[0]
        val = np.matmul(X, w) + svm.intercept_[0]
        labeled = np.sign(val)
        return np.sum(labeled == np.ravel(y.T)) / X.shape[0]

    def get_perc_accu(self, perc_w, X, y):
        X_1 = np.c_[X, np.ones(X.shape[0])]
        # val = np.array([np.zeros(X.shape[0])])
        val = np.matmul(X_1, perc_w)
        labeled = np.sign(val)
        return np.sum(labeled == np.ravel(y.T)) / X.shape[0]

    def draw_perc_hyp(self, X, y):
        gc.log("Drawing Perceptron")
        w = self._perc.fit(X.T, np.ravel(y))
        a = -w[0] / w[1]
        xx = np.linspace(-2.5, 2.5)
        yy = a * xx - (w[-1] / w[1])
        plt.plot(xx, yy, label='Perceptron', color='green')

    def draw_true_hyp(self):
        w = np.array([[0.3], [-0.5], [0.1]])
        a = -w[0] / w[1]
        xx = np.linspace(-2.5, 2.5)
        yy = a * xx - (w[-1] / w[1])
        plt.plot(xx, yy, label='Real', color='black')

    def init_plot(self, m):
        fig = plt.figure()
        fig.suptitle("SVM vs Perceptron, " + str(m) + " Samples")
        plt.xlabel('x Coordinate')
        plt.ylabel('y Coordinate')

    def compare_one(self, m):
        gc.log("Comparing one")
        self.init_plot(m)
        points = self.draw_m_points(m)
        raw_labels = np.array([[self.true_label(x, y) for x, y in points.T]])
        labels = np.vstack((points, raw_labels)).T
        good_points = labels[labels[:, 2] == 1]
        bad_points = labels[labels[:, 2] != 1]
        # good_points = labels[bool_labels]
        # bad_points = labels[not bool_labels]
        plt.scatter(good_points[:, 0], good_points[:, 1], label='True',
                    marker='x')
        plt.scatter(bad_points[:, 0], bad_points[:, 1], label='False',
                    marker='x')
        self.draw_svm_hyp(points, raw_labels)
        self.draw_perc_hyp(points, raw_labels)
        self.draw_true_hyp()
        plt.legend()
        self.plot_to_file('svm_vs_perc_' + str(m))

    def compare_many(self):
        gc.log("Comparing many")
        for m in [5, 10, 15, 25, 70]:
            self.compare_one(m)

    def big_test(self):
        gc.log("Big test")
        fig = plt.figure()
        fig.suptitle("SVM vs Perceptron, Accuracy Test")
        plt.xlabel('Train Set Size')
        plt.ylabel('Accuracy (%)')

        k, n_iter = 10000, 500
        M, accurs = [5, 10, 15, 25, 70], [[],[]]
        for m in M:
            svm_accu_sum = 0
            perc_accu_sum = 0
            for i in range(n_iter):
                while True:
                    train_X = self.draw_m_points(m).T
                    train_y = self.true_label(train_X)
                    if np.unique(train_y).shape[0] == 2:
                        break
                while True:
                    test_X = self.draw_m_points(k).T
                    test_y = self.true_label(test_X)
                    if np.unique(test_y).shape[0] == 2:
                        break
                svm = self._svm.fit(train_X, np.ravel(train_y.T))
                perc_w = self._perc.fit(train_X, train_y)
                svm_accu = self.get_svm_accu(svm, test_X, test_y)
                perc_accu = self.get_perc_accu(perc_w, test_X, test_y)
                svm_accu_sum += svm_accu
                perc_accu_sum += perc_accu
            svm_accu_avg = svm_accu_sum / n_iter
            perc_accu_avg = perc_accu_sum / n_iter
            accurs[0].append(svm_accu_avg)
            accurs[1].append(perc_accu_avg)
        plt.plot(M, accurs[0], label = 'SVM')
        plt.plot(M, accurs[1], label='Perceptron')
        plt.legend()
        self.plot_to_file('q5',2)