"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
from ex4_tools import DecisionStump, decision_boundaries, generate_data, \
    load_images
import matplotlib.pyplot as plt
from adaboost import AdaBoost
import comparer as cmp
from face_detection import integral_image, WeakImageClassifier


def Q4():
    comp = cmp.Comparer()
    comp.compare_many()
    'TODO complete this function'


def Q5():
    comp = cmp.Comparer()
    comp.big_test()
    'TODO complete this function'


def Q8():
    n_samples, noise,T = 5000, 0,500
    X, y = generate_data(n_samples, noise)
    D = np.array([1.0 / n_samples] * n_samples)
    WL = DecisionStump(D, X, y)
    ada  = AdaBoost(WL, T)
    ada.train(X, y)
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    test_n_samples, test_noise = 200, 0
    test_X, test_y = generate_data(test_n_samples, test_noise)
    train_err = np.apply_along_axis(ada.error, 0, np.array([np.arange(T)]),
                                    X, y)
    # train_err = ada.error(X,y, np.arange(T))
    # fig = plt.plot()
    # for t in range(T):
    #     train_err = ada.error(X, y, t)
    #     plt.plot(np.arange(T),train_err)
    #     # test_err = ada.get_test_err(test_X,test_y)


    'TODO complete this function'


def Q9():
    'TODO complete this function'


def Q10():
    'TODO complete this function'


def Q11():
    'TODO complete this function'


def Q12():
    'TODO complete this function'


def Q17():
    'TODO complete this function'


def Q18():
    'TODO complete this function'


if __name__ == '__main__':
    # Q4()
    Q5()
    # Q8()
    'TODO complete this function'
