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
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        m = X.shape[0]
        D = []
        for i in range(self.T):
            r=[]
            for j in range(m):
                r.append(1.0/m)
            D.append(r)

        D = np.array(D)
        # D = np.array([[1.0 / m] * m] * self.T)
        for t in range(self.T - 1):
            self.WL.train(D[t], X, y)
            self.h[t] = self.WL.predict
            gc.log("X is \n", X.shape, X)
            prediction = self.h[t](X)
            gc.log("Prediction is \n", prediction.shape, prediction)
            gc.log("True is \n", y.shape, y)
            indicator = (prediction != y)
            gc.log("Indicator is \n", indicator.shape, indicator)
            gc.log("Num of wrong = ", np.count_nonzero(indicator))
            eps = np.sum(D[t][indicator])
            gc.log("eps is \n", eps)
            self.w[t] = (0.5) * np.log((1.0-eps) / float(eps))
            gc.log('W is \n', self.w[:t+1])
            expo = np.exp(-self.w[t] * (prediction * y))
            gc.log("expo is \n", expo.shape, expo)
            D[t + 1] = D[t] * expo
            gc.log("D:")
            gc.log(D[t])
            D[t + 1] /= np.sum(D[t + 1])
            gc.log(D[t+1])
            gc.log("Sum of D: ", np.sum(D[t+1]))
            print()
            print()

        self.h[self.T] = self.WL.predict
        return self.w[self.T - 1]

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
        gc.log('W is\n', self.w.shape, self.w)
        gc.log('X is\n', X.shape, X)
        gc.log('T is ', max_t)
        return np.sign(np.sum([self.w[i] * self.h[i](X) for i in range(max_t)]))
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
