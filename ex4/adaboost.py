"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


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
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = X.shape[0]
        D = [[1.0 / m] * m] * self.T
        for t in range(self.T - 1):
            self.h[t] = self.WL(D, X, y)
            prediction = self.h[t].predict(X)
            indicator = prediction == y
            eps = np.sum(D[t][indicator])
            self.w[t] = (0.5) * np.log((1.0 / eps) - 1)
            expo = np.exp(-self.w[t] * (prediction * y))
            D[t + 1] = D[t] * expo
            D[t + 1] /= np.sum(D[t + 1])

        return self.w[self.T]

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
        return np.sign(np.sum([self.w[:max_t] * self.h[i](X)] for i in range(
                max_t)))
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


        # TODO complete this function


