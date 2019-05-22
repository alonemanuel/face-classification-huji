import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import garcon as gc


class Perceptron:
    def __init__(self):
        self._X_train = None
        self._y_train = None
        self._curr_w = None
        self._inner_vec = None
        self._signs = None

    def init_weights(self, size):
        self._curr_w = np.zeros([size, 1])

    def get_inner(self):
        self._inner_vec = np.matmul(self._X_train, self._curr_w)

    def get_signs(self):
        self._signs = np.sign(self._inner_vec).T
        # self._signs =  self._y_train.T* self._inner_vec

    def check_and_update(self):

        bad_idxs = np.where(self._signs[0] != self._y_train[0])[0]
        if bad_idxs.shape[0] == 0:
            return True
        else:
            # If there are bad indices, we should update and return false.
            some_idx = bad_idxs[0]
            self._curr_w += self._y_train[0][some_idx] * np.array([
                self._X_train[
                    some_idx]]).T
            return False

    def fit(self, X, y):
        '''
        :param X: shape: (n_samples, n_features)
        :param y: shape: (n_samples,1)
        :return:
        '''
        X_1 = np.c_[X, np.ones(X.shape[0])]
        w = np.zeros(X_1.shape[1])
        while True:
            signs = np.sign(np.matmul(X_1, w))
            comp_idxs = np.where(signs != np.ravel(y.T))[0]
            # print('Comp:')
            # print(np.ravel(y.T))
            if comp_idxs.shape[0] == 0:
                return w
            w += y[comp_idxs[0], 0] * X_1[comp_idxs[0]]


    def predict(self, x):
        # The real result
        real_res = np.inner(x, self._curr_w)
        return np.sign(real_res)
