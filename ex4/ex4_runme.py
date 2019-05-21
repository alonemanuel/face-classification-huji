"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
from ex4.ex4_tools import DecisionStump, decision_boundaries, generate_data, \
    load_images
import matplotlib.pyplot as plt
from ex4.adaboost import AdaBoost
from ex4.perceptron impoty Perceptron
from ex4.face_detection import integral_image, WeakImageClassifier


def Q4():
    perc
    'TODO complete this function'


def Q5():
    'TODO complete this function'


def Q8():
    n_samples, noise = 5000, 0
    X, y = generate_data(n_samples, noise)
    D = np.array([1.0 / n_samples] * n_samples)
    WL, T = DecisionStump(D, X, y), np.arange(500)
    ada = [AdaBoost(WL, t) for t in T]
    test_n_samples, test_noise = 200, 0
    test_X, test_y = generate_data(test_n_samples, test_noise)
    training_err =
    test_err =
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
    Q4()
    Q5()
    Q8()
    'TODO complete this function'
