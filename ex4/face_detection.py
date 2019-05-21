"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton the weak image classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


def integral_image(images):
    '''
    compute the integral of the images
    :param images: numpy array of images, shape=(num_samples, image_height, image_width)
    :return: numpy array of the integrals of the input, the same shape
    '''
    # TODO complete this function


def sum_square(integrals, up, left, height, width):
    '''
    compute the sum of the pixels in the square between the upper left pixel (up, left)
    and down right pixel (up + height - 1, left + width - 1). include the corners in the square.
    :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
    :param up: the up limit of the square
    :param left: the left limit of the square
    :param height: the height of the square
    :param width: the width of the square
    :return: the sum of the pixels in the square (int)
    '''
    # TODO complete this function


class WeakImageClassifier:

    def __init__(self, sample_weight, integrals, labels):
        """
        Train the classifier over the sample (integrals,labels) w.r.t. the weights sample_weight over integrals

        Parameters
        ----------
        sample_weight : weights over the sample numpy array, shape=(num_samples)
        integrals: numpy array shape=(num_samples, image_height, image_width), the samples
        labels: numpy array shape=(num_samples)
        """
        _, self.rows, self.cols = integrals.shape
        self.up = 0
        self.height = 0
        self.left = 0
        self.width = 0
        self.theta = np.inf
        self.loss = np.inf
        self.sign = 0
        self.kernel = None
        self.train(integrals, labels, sample_weight)

    def kernel_a(self, integrals, up, left, height, width):
        '''
        calculate the value of Haar feature of type A. the white part is located between the upper left pixel (up, left)
        and down right pixel (up + height - 1, left + width - 1). the black part located in its right side.
        :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
        :param up: the up limit of the square
        :param left: the left limit of the square
        :param height: the height of the square
        :param width: the width of the square
        :return: the values of the Haar feature for each pixel- numpy array shape=(num_samples)
        '''
        sum_left_square = sum_square(integrals, up, left, height, width)
        sum_right_square = sum_square(integrals, up, left + width, height, width)
        feature_values = sum_left_square - sum_right_square
        return feature_values

    def kernel_b(self, integrals, up, left, height, width):
        '''
        calculate the value of Haar feature of type B. the white part is located between the upper left pixel (up, left)
        and down right pixel (up + height - 1, left + width - 1). the black part located right below.
        :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
        :param up: the up limit of the square
        :param left: the left limit of the square
        :param height: the height of the square
        :param width: the width of the square
        :return: the values of the Haar feature for each pixel- numpy array shape=(num_samples)
        '''
        # TODO complete this function

    def kernel_c(self, integrals, up, left, height, width):
        '''
        calculate the value of Haar feature of type C. the first white part is located between the upper left pixel
        (up, left) and down right pixel (up + height - 1, left + width - 1). the black part located in its right side
        and the second right part located in the blacks part side.
        :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
        :param up: the up limit of the square
        :param left: the left limit of the square
        :param height: the height of the square
        :param width: the width of the square
        :return: the values of the Haar feature for each pixel- numpy array shape=(num_samples)
        '''
        # TODO complete this function

    def kernel_d(self, integrals, up, left, height, width):
        '''
        calculate the value of Haar feature of type C. the first white part is located between the upper left pixel
        (up, left) and down right pixel (up + height - 1, left + width - 1). the first black parts located in its right
        side, and in it's bottom. the second white part located in its bottom right
        :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
        :param up: the up limit of the square
        :param left: the left limit of the square
        :param height: the height of the square
        :param width: the width of the square
        :return: the values of the Haar feature for each pixel- numpy array shape=(num_samples)
        '''
        # TODO complete this function

    def evaluate_kernel(self, integrals, labels, kernel, up, left, height, width, weights):
        '''
        Get the feature values according to the following parameters. Try the hypothesis of threshold classifier over
        the feature values. the threshold can be either of type > or of type <.
        :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
        :param labels: the labels of the samples shape=(num_samples)
        :param kernel: An Haar feature function of the type {a,b,c,d}.
        :param up: the up limit of the square
        :param left: the left limit of the square
        :param height: the height of the square
        :param width: the width of the square
        :param weights: the current weight of the samples shape=(num_samples)
        '''
        # TODO complete this function

    def evaluate_all_kernel_types(self, integrals, weights, labels, up, left, height, width):
        '''
        For each of the {a,b,c,d} kernel functions, if the following parameters are legal, Try the hypothesis of
        threshold classifier over the feature values.
        :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
        :param weights: the current weight of the samples, shape=(num_samples)
        :param labels: the labels of the samples, shape=(num_samples)
        :param up: the up limit of the square
        :param left: the left limit of the square
        :param height: the height of the square
        :param width: the width of the square
        '''
        if up + height <= self.rows and left + 2 * width <= self.cols:
            self.evaluate_kernel(integrals, labels, self.kernel_a, up, left, height, width, weights)

        if up + 2 * height <= self.rows and left + width <= self.cols:
            self.evaluate_kernel(integrals, labels, self.kernel_b, up, left, height, width, weights)

        if up + height <= self.rows and left + 3 * width <= self.cols:
            self.evaluate_kernel(integrals, labels, self.kernel_c, up, left, height, width, weights)

        if up + 2 * height <= self.rows and left + 2 * width <= self.cols:
            self.evaluate_kernel(integrals, labels, self.kernel_d, up, left, height, width, weights)

    def evaluate_feature_performance(self, kernel, feature_values, weights, labels, up, height, left, width, sign):
        '''
        find the best decision stump hypothesis for given feature value, and update parameters accordingly.
        For given feature values and labels find the ERM for the threshold problem, if the loss value according some
        theta is lower than self.loss update the parameters of self to the parameters of the function and update theta
        to the best theta.
        :param kernel: function- 'kernel_k' (where k in {a,b,c,d})
        :param feature_values: the feature value of the image according to the kernel configured by the following parameters
        :param weights: the of of the samples, shape=(num_samples)
        :param labels: the labels of the data, shape=(num_samples)
        :param up: the up limit of the square
        :param height: the height of the square
        :param left: the left limit of the square
        :param width: the width of the square
        :param sign: whether the upper-left square of the kernel is white or black (equivalent to multiply the feature
        by 1 or -1.
        '''
        # TODO complete this function

    def train(self, integrals, labels, sample_weight):
        '''
        This function iterate over all possible Haar features (of the 4 types we defined) and find the best hypothesis
        for the current distribution (ERM)
        :param integrals: the integrals of the images in the dataset, shape=(num_samples, image_height, image_width)
        :param labels: the labels of the samples in the dataset, shape=(num_samples)
        :param sample_weight: the current weights of the samples. shape=(num_samples)
        '''
        num_samples, self.rows, self.cols = integrals.shape
        for up in range(self.rows):
            for height in range(1, self.rows + 1):
                for left in range(self.cols):
                    for width in range(1, self.cols + 1):
                        self.evaluate_all_kernel_types(integrals, sample_weight, labels, up, left, height, width)

    def predict(self, integrals):
        '''
        predict labels (whether the image contain face or not) for the images according to their integrals.
        :param integrals: the integrals of the images we want to predict.
        :return: labels of the images
        '''
        # TODO complete this function

    def visualize_kernel(self):
        '''
        This function visualize the kernel.
        :return: image of the kernel
        '''
        image = np.zeros((self.rows, self.cols))
        if self.kernel == self.kernel_a:
            image[self.up: self.up + self.height, self.left: self.left + self.width] = 1
            image[self.up: self.up + self.height, self.left + self.width: self.left + self.width * 2] = -1
        if self.kernel == self.kernel_b:
            image[self.up: self.up + self.height, self.left: self.left + self.width] = 1
            image[self.up + self.height: self.up + self.height * 2, self.left: self.left + self.width] = -1
        if self.kernel == self.kernel_c:
            image[self.up: self.up + self.height, self.left: self.left + self.width] = 1
            image[self.up: self.up + self.height, self.left + self.width: self.left + self.width * 2] = -1
            image[self.up: self.up + self.height, self.left + self.width * 2: self.left + self.width * 3] = 1
        if self.kernel == self.kernel_d:
            image[self.up: self.up + self.height, self.left: self.left + self.width] = 1
            image[self.up: self.up + self.height, self.left + self.width: self.left + self.width * 2] = -1
            image[self.up + self.height: self.up + self.height * 2, self.left: self.left + self.width] = -1
            image[self.up + self.height: self.up + self.height * 2, self.left + self.width: self.left + self.width * 2] = 1
        return image * self.sign

