import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import garcon as gc


class perceptron:
    def __init__(self):
        gc.log("Creating perceptron")
        self._X_train = None
        self._y_train = None
        self._curr_w = None
        self._inner_vec = None
        self._signs = None

    def init_weights(self, size):
        self._curr_w = np.zeros([size, 1])

    def get_inner(self):
        gc.log("Getting inner")
        self._inner_vec = np.matmul(self._X_train, self._curr_w)

    def get_signs(self):
        self._signs = np.inner(self._inner_vec, self._y_train)

    def check_and_update(self):
        # Vector of indices that point to signs that are smaller or equal to
        # zero. These are considered "bad indices".
        bad_idxs = np.where(self._signs <= 0)[0]
        # If there are no bad indices, we succeeded.
        if bad_idxs.shape[0] == 0:
            return True
        else:
            # If there are bad indices, we should update and return false.
            some_idx = bad_idxs[0]
            self._curr_w += self._y_train[some_idx] * self._X_train[some_idx]
            return False

    def fit(self, X, y):
        gc.log("Fitting")
        # Init X and y vecs with place for the bias
        self._X_train = np.c_[X, np.ones(X.shape[0])]
        self._y_train = y.append(0)
        self.init_weights(X.shape[1])
        while True:
            # TODO: Plot perceptron converging
            self.get_inner()
            self.get_signs()
            if self.check_and_update():
                return

    def predict(self, x):
        gc.log("Predicting")
        # The real result
        real_res = np.inner(x, self._curr_w)
        return np.sign(real_res)
