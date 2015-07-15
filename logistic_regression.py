__author__ = 'Henry'

import numpy as np
from matplotlib import pyplot as plt

class LogisticRegression:
	X = np.array([])
	Y = np.array([])
	W = np.array([])

	def __init__(self):
		pass

	def fit(self, dataset, class_index, skip_header=True):
		if isinstance(dataset, str):
			dataset = np.genfromtxt(dataset, delimiter=',', skip_header=skip_header, dtype=np.str)

		if isinstance(dataset, np.ndarray):
			self.Y = dataset[:, class_index]
			self.X = np.delete(dataset, class_index, axis=1).astype(np.float32)

		W = np.array([])
		return self

	def train(self, alpha, threshold=0.1):
		current_error = threshold * 3
		self.W = np.ones(self.X.shape[1] + 1).astype(np.float32)
		X = np.hstack((np.ones((self.X.shape[0], 1)), self.X)).astype(np.float32)
		Y = LogisticRegression.__string_to_int__(self.Y).astype(np.float32)
		N = self.X.shape[0]
		_W = np.array(self.W)

		while not (np.any(np.isnan(_W) == True) or np.any(np.isinf(_W) == True)):
			_W = np.array(self.W)
			last_error = current_error
			H = self.__predictive_model__(X, _W)
			current_error = self.__cost_function__(N, H, Y)

			if np.abs(last_error - current_error) < threshold and not np.isinf(current_error):
				return self

			for i, w in enumerate(self.W):
				_W[i] = w - (float(alpha) * LogisticRegression.__derived__(i, N, self.W, H, Y))
			self.W = _W

	@staticmethod
	def __derived__(i, N, W, H, Y):
		if i == 0:
			return LogisticRegression.__derived_theta_zero__(N, W, H, Y)
		return LogisticRegression.__derived_theta_one__(N, W, H, Y)

	@staticmethod
	def __derived_theta_zero__(N, W, X, Y):
		result = 0.0
		for i in range(0, Y.shape[0]):
			result += (1.0 - (2.0 * Y[i])) / ((2.0 * N * np.e**(W[0] + W[1] * X[i])) + (2.0 * N))
		return result

	@staticmethod
	def __derived_theta_one__(N, W, X, Y):
		result = 0.0
		for i in range(0, Y.shape[0]):
			result += ((X[i] - (2.0 * X[i] * Y[i])) / ((2.0 * N * np.e**(W[0] + (W[1] * X[i]))) + (2.0 * N)))
		return result

	@staticmethod
	def __cost_function__(N, H, Y):
		return (-1.0/float(N)) * LogisticRegression.__lazy_cost_function__(H, Y)

	@staticmethod
	def __lazy_cost_function__(H, Y):
		result = 0.0
		for i in range(0, Y.shape[0]):
			a = np.nan_to_num(np.log2(H[i]) * Y[i])
			b = np.nan_to_num((1. - Y[i]) * np.log2((1. - H[i])))
			result += a + b
		return result

	@staticmethod
	def __predictive_model__(X, W):
		return np.round(1.0/(1.0 + np.power(np.e, -(np.dot(W, X.T)))))

	@staticmethod
	def __string_to_int__(Y):
		if np.all(Y == np.float32):
			values = list(set(Y))
			values = dict(zip(values, range(0, len(values))))
			_Y = np.array([values[y] for y in Y])
			return _Y
		return Y

	def test(self, X):
		X = np.vstack((np.ones(X.shape[0]), X)).T
		return LogisticRegression.__predictive_model__(X, self.W)

	def plot(self, model_X, model_Y):
		plt.scatter(self.X, self.Y, label='f(x)')

		if isinstance(model_Y, list) or isinstance(model_Y, np.ndarray):
			plt.plot(model_X, model_Y, label='h(x)')
		elif isinstance(model_Y, dict):
			for y in model_Y.items():
				plt.plot(model_X, y[1], label=y[0])
		plt.legend()
		plt.show()

def main():
	"""
	Limitations:
	Binary datasets only!
	One predictive attribute only!
	"""
	model = LogisticRegression()

	dataset = np.array([
		[5., 1],
	    [2., 1],
		[1., 1],
		[3., 0],
		[4., 0]
	]).astype(np.float32)

	model.fit(dataset, skip_header=True, class_index=-1)
	model.train(alpha=100., threshold=1.)

	x_axis = np.arange(np.min(model.X), np.max(model.X), 0.1)
	y_axis = model.test(x_axis)
	model.plot(x_axis, y_axis)
	print model.W
	z = 0


main()