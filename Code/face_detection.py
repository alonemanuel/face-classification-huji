# """
# ===================================================
#      Introduction to Machine Learning (67577)
# ===================================================
#
# Skeleton the weak image classifier.
#
# Author: Gad Zalcberg
# Date: February, 2019
#
# """
# import numpy as np
# import matplotlib.pyplot
#
# def integral_image(images):
#     '''
#     compute the integral of the images
#     :param images: numpy array of images, shape=(num_samples, image_height, image_width)
#     :return: numpy array of the integrals of the input, the same shape
#     '''
#     integral = images.astype(np.float64)
#     num_samples, cols, rows = images.shape
#     for i in range(rows):
#         for j in range(cols):
#             if i > 0:
#                 integral[:, i, j] += integral[:, i - 1, j]
#             if j > 0:
#                 integral[:, i, j] += integral[:, i, j - 1]
#             if i > 0 and j > 0:
#                 integral[:, i, j] -= integral[:, i - 1, j - 1]
#     return integral
#
#
# def sum_square(integrals, up, left, height, width):
#     '''
#     compute the sum of the pixels in the square between the upper left pixel (up, left)
#     and down right pixel (up + height - 1, left + width - 1). include the corners in the square.
#     :param integfcopyrals: the integrals of the images, shape=(num_samples,
#     image_height, image_width)
#     :param up: the up limit of the square
#     :param left: the left limit of the square
#     :param height: the height of the square
#     :param width: the width of the square
#     :return: the sum of the pixels in the square
#     '''
#     sum = np.copy(integrals[:, up + height - 1, left + width - 1])
#     if up > 0:
#         sum -= integrals[:, up - 1, left]
#     if left > 0:
#         sum -= integrals[:, up, left - 1]
#     if up > 0 and left > 0:
#         sum += integrals[:, up - 1, left - 1]
#     return sum
#
#
# class WeakImageClassifier:
#
#     def __init__(self, sample_weight, integrals, labels):
#         """
#         Train the classifier over the sample (integrals,labels) w.r.t. the weights sample_weight over integrals
#
#         Parameters
#         ----------
#         sample_weight : weights over the sample
#         integrals, labels: samples
#         """
#         _, self.rows, self.cols = integrals.shape
#         self.up = 0
#         self.height = 0
#         self.left = 0
#         self.width = 0
#         self.theta = np.inf
#         self.loss = np.inf
#         self.sign = 0
#         self.kernel = None
#         self.train(integrals, labels, sample_weight)
#
#     def kernel_a(self, integrals, up, left, height, width):
#         '''
#         calculate the value of Haar feature of type A. the white part is located between the upper left pixel (up, left)
#         and down right pixel (up + height - 1, left + width - 1). the black part located in its right side.
#         :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
#         :param up: the up limit of the square
#         :param left: the left limit of the square
#         :param height: the height of the square
#         :param width: the width of the square
#         :return: the values of the Haar feature
#         '''
#         sum_left_square = sum_square(integrals, up, left, height, width)
#         sum_right_square = sum_square(integrals, up, left + width, height, width)
#         feature_values = sum_left_square - sum_right_square
#         return feature_values
#
#     def kernel_b(self, integrals, up, left, height, width):
#         '''
#         calculate the value of Haar feature of type B. the white part is located between the upper left pixel (up, left)
#         and down right pixel (up + height - 1, left + width - 1). the black part located right below.
#         :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
#         :param up: the up limit of the square
#         :param left: the left limit of the square
#         :param height: the height of the square
#         :param width: the width of the square
#         :return: the values of the Haar feature
#         '''
#         sum_up_square = sum_square(integrals, up, left, height, width)
#         sum_down_square = sum_square(integrals, up + height, left, height, width)
#         feature_values = sum_down_square - sum_up_square
#         return feature_values
#
#     def kernel_c(self, integrals, up, left, height, width):
#         '''
#         calculate the value of Haar feature of type C. the first white part is located between the upper left pixel
#         (up, left) and down right pixel (up + height - 1, left + width - 1). the black part located in its right side
#         and the second right part located in the blacks part side.
#         :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
#         :param up: the up limit of the square
#         :param left: the left limit of the square
#         :param height: the height of the square
#         :param width: the width of the square
#         :return: the values of the Haar feature
#         '''
#         sum_left_square = sum_square(integrals, up, left, height, width)
#         sum_middle_square = sum_square(integrals, up, left + width, height, width)
#         sum_right_square = sum_square(integrals, up, left + 2 * width, height, width)
#         feature_values = sum_left_square - sum_middle_square + sum_right_square
#         return feature_values
#
#     def kernel_d(self, integrals, up, left, height, width):
#         '''
#         calculate the value of Haar feature of type C. the first white part is located between the upper left pixel
#         (up, left) and down right pixel (up + height - 1, left + width - 1). the first black parts located in its right
#         side, and in it's bottom. the second white part located in its bottom right
#         :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
#         :param up: the up limit of the square
#         :param left: the left limit of the square
#         :param height: the height of the square
#         :param width: the width of the square
#         :return: the values of the Haar feature
#         '''
#         sum_up_left_square = sum_square(integrals, up, left, height, width)
#         sum_down_left_square = sum_square(integrals, up + height, left, height, width)
#         sum_up_right_square = sum_square(integrals, up, left + width, height, width)
#         sum_down_right_square = sum_square(integrals, up + height, left + width, height, width)
#         feature_values = sum_up_left_square - sum_up_right_square - sum_down_left_square + sum_down_right_square
#         return feature_values
#
#     def evaluate_kernel(self, integrals, labels, kernel, up, left, height, width, weights):
#         '''
#         Get the feature values according to the following parameters. Try the hypothesis of threshold classifier over
#         the feature values. the threshold can be either of type > or of type <.
#         :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
#         :param labels: the labels of the samples
#         :param kernel: An Haar feature function of the type {a,b,c,d}.
#         :param up: the up limit of the square
#         :param left: the left limit of the square
#         :param height: the height of the square
#         :param width: the width of the square
#         :param weights: the current weight of the samples
#         '''
#         feature_values = kernel(integrals, up, left, height, width)
#         self.evaluate_feature_performance(kernel, feature_values, weights, labels, up, height, left, width, sign=1)
#         self.evaluate_feature_performance(kernel, feature_values, weights, labels, up, height, left, width, sign=-1)
#
#     def evaluate_all_kernel_types(self, integrals, weights, labels, up, left, height, width):
#         '''
#         For each of the {a,b,c,d} kernel functions, if the following parameters are legal, Try the hypothesis of
#         threshold classifier over the feature values.
#         :param integrals: the integrals of the images, shape=(num_samples, image_height, image_width)
#         :param weights: the current weight of the samples
#         :param labels: the labels of the samples
#         :param up: the up limit of the square
#         :param left: the left limit of the square
#         :param height: the height of the square
#         :param width: the width of the square
#         '''
#         if up + height <= self.rows and left + 2 * width <= self.cols:
#             self.evaluate_kernel(integrals, labels, self.kernel_a, up, left, height, width, weights)
#
#         if up + 2 * height <= self.rows and left + width <= self.cols:
#             self.evaluate_kernel(integrals, labels, self.kernel_b, up, left, height, width, weights)
#
#         if up + height <= self.rows and left + 3 * width <= self.cols:
#             self.evaluate_kernel(integrals, labels, self.kernel_c, up, left, height, width, weights)
#
#         if up + 2 * height <= self.rows and left + 2 * width <= self.cols:
#             self.evaluate_kernel(integrals, labels, self.kernel_d, up, left, height, width, weights)
#
#     def evaluate_feature_performance(self, kernel, feature_values, weights, labels, up, height, left, width, sign):
#         '''
#         find the best decision stump hypothesis for given feature value, and update parameters accordingly.
#         for given feature values and labels find the ERM for the threshold problem, if the loss value according some
#         theta is lower than self.loss update the parameters of self to the parameters of the function and update theta
#         to the best theta.
#         :param kernel: function- 'kernel_k' for k in {a,b,c,d}
#         :param feature_values: the feature value of the image according to the kernel configured by the following parameters
#         :param weights: the of of the samples
#         :param labels: the labels of the data
#         :param up: the up limit of the square
#         :param height: the height of the square
#         :param left: the left limit of the square
#         :param width: the width of the square
#         :param sign: whether the upper-left square of the kernel is white or black (equivalent to multiply the feature
#         by 1 or -1.
#         '''
#         feature_values_signed = feature_values * sign
#
#         sort_indices = feature_values_signed.argsort()
#         sorted_samples = feature_values_signed[sort_indices]
#         sorted_weights = weights[sort_indices]
#         sorted_labels = labels[sort_indices]
#
#         minimal_theta_loss = np.sum(sorted_weights[sorted_labels > 0])
#         losses = np.append(minimal_theta_loss, minimal_theta_loss - np.cumsum(sorted_weights * sorted_labels))
#         thetas = np.concatenate([[-np.inf], (sorted_samples[1:] + sorted_samples[:-1]) / 2, [np.inf]])
#         min_loss_idx = np.argmin(losses)
#         if self.loss > losses[min_loss_idx]:
#             self.up = up
#             self.height = height
#             self.left = left
#             self.width = width
#             self.theta = thetas[min_loss_idx]
#             self.loss = losses[min_loss_idx]
#             self.sign = sign
#             self.kernel = kernel
#
#     def train(self, integrals, labels, sample_weight):
#         '''
#         This function iterate over all possible Haar features (of the 4 types we defined) and find the best hypothesis
#         for the current distribution (ERM)
#         :param integrals: the integrals of the images in the dataset, shape=(num_samples, image_height, image_width)
#         :param labels: the labels of the samples in the dataset
#         :param sample_weight: the current weights of the samples.
#         '''
#         num_samples, self.rows, self.cols = integrals.shape
#         for up in range(self.rows):
#             for height in range(1, self.rows + 1):
#                 for left in range(self.cols):
#                     for width in range(1, self.cols + 1):
#                         self.evaluate_all_kernel_types(integrals, sample_weight, labels, up, left, height, width)
#         plt.imshow(self.visualize_kernel())
#         plt.show()
#     def predict(self, integrals):
#         '''
#         predict labels (whether the image contain face or not) for the images according to their integrals.
#         :param integrals: the integrals of the images we want to predict.
#         :return: labels of the images
#         '''
#         feature_values = self.kernel(integrals, self.up, self.left, self.height, self.width)
#         feature_values *= self.sign
#         labels = np.zeros(len(feature_values))
#         labels[feature_values >= self.theta] = -1
#         labels[feature_values < self.theta] = 1
#         return labels
#
#     def visualize_kernel(self):
#         '''
#         This function visualize the kernel.
#         :return: image of the kernel
#         '''
#         image = np.zeros((self.rows, self.cols))
#         if self.kernel == self.kernel_a:
#             image[self.up: self.up + self.height, self.left: self.left + self.width] = 1
#             image[self.up: self.up + self.height, self.left + self.width: self.left + self.width * 2] = -1
#         if self.kernel == self.kernel_b:
#             image[self.up: self.up + self.height, self.left: self.left + self.width] = 1
#             image[self.up + self.height: self.up + self.height * 2, self.left: self.left + self.width] = -1
#         if self.kernel == self.kernel_c:
#             image[self.up: self.up + self.height, self.left: self.left + self.width] = 1
#             image[self.up: self.up + self.height, self.left + self.width: self.left + self.width * 2] = -1
#             image[self.up: self.up + self.height, self.left + self.width * 2: self.left + self.width * 3] = 1
#         if self.kernel == self.kernel_d:
#             image[self.up: self.up + self.height, self.left: self.left + self.width] = 1
#             image[self.up: self.up + self.height, self.left + self.width: self.left + self.width * 2] = -1
#             image[self.up + self.height: self.up + self.height * 2, self.left: self.left + self.width] = -1
#             image[self.up + self.height: self.up + self.height * 2, self.left + self.width: self.left + self.width * 2] = 1
#         return image * self.sign
#
#
#
#


"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton the weak image classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np
import garcon as gc
import matplotlib.pyplot as plt
from ex4_tools import find_threshold

def S(integrals, a, b):
	'''
	Compute the integral value of the (a,b) cell in images.
	:param integrals: integrals of some image, shape=(num_samples,
	image_height, image_width)
	:param a: row idx to calculate, shape=(0)
	:param b: col idx to calculate, shape=(0)
	:return: A vector of the integral value of (a,b), shape=(num_samples,0)
	'''
	if a >= 0 and b >= 0:
		if a < integrals.shape[1] and b < integrals.shape[2]:
			return np.copy(integrals[:, a, b])
	return 0.0

def integral_image(images):
	'''
	compute the integral of the images
	:param images: numpy array of images, shape=(num_samples, image_height, image_width)
	:return: numpy array of the integrals of the input, the same shape
	'''
	integral = np.zeros(images.shape)
	for a in range(integral.shape[1]):
		for b in range(integral.shape[2]):
			sum = np.copy(images[:, a, b])
			sum += S(integral, a - 1, b)
			sum += S(integral, a, b - 1)
			sum -= S(integral, a - 1, b - 1)
			integral[:, a, b] = sum

	return integral

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
	dr_a, dr_b = up + height - 1, left + width - 1
	dl_a, dl_b = up + height - 1, left
	ur_a, ur_b = up, left + width - 1
	ul_a, ul_b = up, left

	sum = S(integrals, dr_a, dr_b)
	sum -= S(integrals, ur_a - 1, ur_b)
	sum -= S(integrals, dl_a, dl_b - 1)
	sum += S(integrals, ul_a - 1, ul_b - 1)

	return sum

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
		sum_right_square = sum_square(integrals, up, left + width, height,
									  width)
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
		sum_upper_rec = sum_square(integrals, up, left, height, width)
		sum_lower_rec = sum_square(integrals, up + height, left, height, width)
		feature_values = sum_upper_rec - sum_lower_rec
		return feature_values

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
		sum_left_rec = sum_square(integrals, up, left, height, width)
		sum_mid_rec = sum_square(integrals, up, left + width, height, width)
		sum_right_rec = sum_square(integrals, up, left + 2 * width, height,
								   width)

		feature_values = sum_left_rec - sum_mid_rec + sum_right_rec
		return feature_values

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
		sum_topleft_square = sum_square(integrals, up, left, height, width)
		sum_topright_square = sum_square(integrals, up, left + width, height,
										 width)
		sum_bottleft_square = sum_square(integrals, up + height, left, height,
										 width)
		sum_bottright_square = sum_square(integrals, up + height, left + width,
										  height, width)
		feature_values = sum_topleft_square - sum_topright_square \
						 - sum_bottleft_square + sum_bottright_square
		return feature_values

	# TODO complete this function

	def evaluate_kernel(self, integrals, labels, kernel, up, left, height,
						width, weights):
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
		feature_values = kernel(integrals, up, left, height, width)
		self.evaluate_feature_performance(kernel, feature_values, weights,
										  labels, up, height, left, width, 1)
		self.evaluate_feature_performance(kernel, feature_values, weights,
										  labels, up, height, left, width, -1)

	# TODO complete this function

	def evaluate_all_kernel_types(self, integrals, weights, labels, up, left,
								  height, width):
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
			self.evaluate_kernel(integrals, labels, self.kernel_a, up, left,
								 height, width, weights)

		if up + 2 * height <= self.rows and left + width <= self.cols:
			self.evaluate_kernel(integrals, labels, self.kernel_b, up, left,
								 height, width, weights)

		if up + height <= self.rows and left + 3 * width <= self.cols:
			self.evaluate_kernel(integrals, labels, self.kernel_c, up, left,
								 height, width, weights)

		if up + 2 * height <= self.rows and left + 2 * width <= self.cols:
			self.evaluate_kernel(integrals, labels, self.kernel_d, up, left,
								 height, width, weights)

	def evaluate_feature_performance(self, kernel, feature_values, weights,
									 labels, up, height, left, width, sign):
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
		# TODO: Possibly reshape
		# loss, theta = find_threshold(weights, feature_values.reshape((-1, 1)),
		#                              labels,
		#                              sign, 0)
		feature_values_signed = feature_values * sign

		sort_indices = feature_values_signed.argsort()
		sorted_samples = feature_values_signed[sort_indices]
		sorted_weights = weights[sort_indices]
		sorted_labels = labels[sort_indices]

		minimal_theta_loss = np.sum(sorted_weights[sorted_labels > 0])
		losses = np.append(minimal_theta_loss, minimal_theta_loss - np.cumsum(
				sorted_weights * sorted_labels))
		thetas = np.concatenate(
				[[-np.inf], (sorted_samples[1:] + sorted_samples[:-1]) / 2,
				 [np.inf]])
		min_loss_idx = np.argmin(losses)
		loss = losses[min_loss_idx]
		theta = thetas[min_loss_idx]
		if loss < self.loss:
			self.up = up
			self.height = height
			self.left = left
			self.width = width
			self.theta = theta
			self.loss = loss
			self.sign = sign
			self.kernel = kernel

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
						self.evaluate_all_kernel_types(integrals, sample_weight,
													   labels, up, left, height,
													   width)
		plt.imshow(self.visualize_kernel())
		plt.show()

	def predict(self, integrals):
		'''
		predict labels (whether the image contain face or not) for the images according to their integrals.
		:param integrals: the integrals of the images we want to predict.
		:return: labels of the images
		'''
		feature_values = self.kernel(integrals, self.up, self.left, self.height,
									 self.width)
		feature_values *= self.sign
		labels = np.zeros(len(feature_values))
		labels[feature_values >= self.theta] = -1
		labels[feature_values < self.theta] = 1
		return labels
		# TODO complete this function

	def visualize_kernel(self):
		'''
		This function visualize the kernel.
		:return: image of the kernel
		'''
		image = np.zeros((self.rows, self.cols))
		if self.kernel == self.kernel_a:
			image[self.up: self.up + self.height,
			self.left: self.left + self.width] = 1
			image[self.up: self.up + self.height,
			self.left + self.width: self.left + self.width * 2] = -1
		if self.kernel == self.kernel_b:
			image[self.up: self.up + self.height,
			self.left: self.left + self.width] = 1
			image[self.up + self.height: self.up + self.height * 2,
			self.left: self.left + self.width] = -1
		if self.kernel == self.kernel_c:
			image[self.up: self.up + self.height,
			self.left: self.left + self.width] = 1
			image[self.up: self.up + self.height,
			self.left + self.width: self.left + self.width * 2] = -1
			image[self.up: self.up + self.height,
			self.left + self.width * 2: self.left + self.width * 3] = 1
		if self.kernel == self.kernel_d:
			image[self.up: self.up + self.height,
			self.left: self.left + self.width] = 1
			image[self.up: self.up + self.height,
			self.left + self.width: self.left + self.width * 2] = -1
			image[self.up + self.height: self.up + self.height * 2,
			self.left: self.left + self.width] = -1
			image[self.up + self.height: self.up + self.height * 2,
			self.left + self.width: self.left + self.width * 2] = 1
		return image * self.sign
