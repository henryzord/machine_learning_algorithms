__author__ = 'Henry'

import numpy as np
from matplotlib import pyplot as plt

class Perceptron:
	"""
	A perceptron with online updating.
	"""
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

		self.W = np.array([])
		return self

	def train(self, alpha, iterations=100):
		self.W = np.ones(self.X.shape[1] + 1).astype(np.float32)
		X = np.hstack((np.ones((self.X.shape[0], 1)), self.X)).astype(np.float32)
		Y = Perceptron.__string_to_int__(self.Y).astype(np.float32)
		W = np.array(self.W)
		iteration = 0

		while iteration < iterations:
			for i, x in enumerate(X):
				h = Perceptron.__predict__(x, W)
				j = Perceptron.__cost_function__(1, h, Y[i])
				if j > 0:
					W = Perceptron.__update_theta__(alpha, W, h, x, Y[i])
			iteration += 1
		self.W = W

	@staticmethod
	def __predict__(X, W):
		return np.sign(np.dot(X, W))

	@staticmethod
	def __cost_function__(N, H, Y):
		return (1.0/(2.0 * float(N))) * np.sum(np.power(H - Y, 2))

	@staticmethod
	def __update_theta__(alpha, W, h, x, y):
		for i, w in enumerate(W):
			W[i] = w - (alpha * (h - y) * x[i])
		return W

	@staticmethod
	def __activation_function__(x):
		return np.sign(x)

	@staticmethod
	def __string_to_int__(Y):
		if Y.dtype != np.float32:
			values = list(set(Y))
			values = dict(zip(values, [-1, 1]))
			_Y = np.array([values[y] for y in Y])
			return _Y
		return Y

	def test(self, X):
		X = np.vstack((np.ones(X.shape[0]), X)).T
		return Perceptron.__predict__(X, self.W)

	def plot(self, model_X, model_Y):
		plt.scatter(self.X, self.Y, label='f(x)', c='green')

		plt.plot(model_X, np.zeros(model_X.shape[0]), c='black')
		plt.plot(np.zeros(model_X.shape[0]), model_X, c='black')

		if isinstance(model_Y, list) or isinstance(model_Y, np.ndarray):
			plt.plot(model_X, model_Y, label='h(x)', c='red')
		elif isinstance(model_Y, dict):
			for y in model_Y.items():
				plt.plot(model_X, y[1], label=y[0])
		plt.legend()
		plt.show()


def main():
	model = Perceptron()

	dataset = np.array([
		[1, -1],
		[2, -1],
		[0, 1],
		[-1, 1]
	])

	model.fit(dataset, skip_header=True, class_index=-1)
	model.train(alpha=0.01, iterations=100)

	x_axis = np.arange(np.min(model.X), np.max(model.X), 0.1)
	y_axis = model.test(x_axis)
	model.plot(x_axis, y_axis)
	print model.W
	z = 0

main()