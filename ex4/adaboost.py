"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
import garcon as gc


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples, )
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        n_samples = X.shape[0]
        D = np.array([1.0 / n_samples] * n_samples)
        # D = np.array([[1.0 / m] * m] * self.T)
        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            y_hat = self.h[t].predict(X)
            mask = y != y_hat
            epsilon = np.matmul(D, mask)
            self.w[t] = 1 / 2 * np.log((1 - epsilon) / epsilon)
            D *= np.exp(-(y * y_hat * self.w[t]))
            D /= np.sum(D)
        # TODO complete this function

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        predictions = np.array([self.h[t].predict(X) for t in range(max_t)])
        multed = np.matmul(self.w[:max_t], predictions)
        signed = np.sign(multed)
        return signed
        # TODO complete this function

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        n_wrong = np.sum(y_hat != y)
        return n_wrong / X.shape[0]

        # TODO complete this function
