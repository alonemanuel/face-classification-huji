"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author: Gad Zalcberg
Date: February, 2019

"""
FIG_DIR3 = '../Images3/'

import garcon as gc
import time

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
    n_samples_train, n_samples_test, noise, T = 5000, 200, 0, 500
    train_X, train_y = generate_data(n_samples_train, noise)
    test_X, test_y = generate_data(n_samples_test, noise)
    WL = DecisionStump
    ada = AdaBoost(WL, T)
    ada.train(train_X, train_y)
    T_range = np.arange(1, T)
    train_errs = [ada.error(train_X, train_y, t) for t in T_range]
    test_errs = [ada.error(test_X, test_y, t) for t in T_range]

    fig = plt.figure()
    fig.suptitle("Train vs Test error, Adaboost")
    plt.xlabel('# of Hypotheses (T)')
    plt.ylabel('Error rate (%)')
    plt.plot(T_range, train_errs, label='Train Error')
    plt.plot(T_range, test_errs, label='Test Error')
    plt.ylim(top=0.06)
    plt.legend()
    plt.savefig(FIG_DIR3 + 'q8')

    return ada, test_X, test_y
    'TODO complete this function'


def Q9(ada, test_X, test_y):
    # f, axs = plt.subplots(3,2)
    n_classifiers = [5, 10, 50, 100, 200, 500]
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        decision_boundaries(ada, test_X, test_y, n_classifiers[i])
    plt.savefig(FIG_DIR3 + 'q9')

    'TODO complete this function'


def Q10():
    'TODO complete this function'


def Q11():
    'TODO complete this function'


def Q12():
    'TODO complete this function'


def Q17():
    train_images, test_images, train_labels, test_labels = load_images(
            '../Docs/')
    train_images = integral_image(train_images)
    test_images = integral_image(test_images)
    WL, T = WeakImageClassifier, 50
    ada = AdaBoost(WL, T)
    ada.train(train_images, train_labels)
    T_range = np.arange(1, T)
    train_errs = [ada.error(train_images, train_labels, t) for t in T_range]
    test_errs = [ada.error(test_images, test_labels, t) for t in T_range]

    fig = plt.figure()
    fig.suptitle("Train vs Test error, Face Classifier")
    plt.xlabel('# of Hypotheses (T)')
    plt.ylabel('Error rate (%)')
    plt.plot(T_range, train_errs, label='Train Error')
    plt.plot(T_range, test_errs, label='Test Error')
    # plt.ylim(top=0.06)
    plt.legend()
    plt.savefig(FIG_DIR3 + 'q17')
    'TODO complete this function'


def Q18():
    'TODO complete this function'


if __name__ == '__main__':
    start_time = time.time()
    # Q4()
    # Q5()
    # learner, test_X, test_y = Q8()
    # Q9(learner, test_X, test_y)
    # Q10()
    Q17()
    gc.log('Execution took %s seconds' % (time.time() - start_time))
    'TODO complete this function'
