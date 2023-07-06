from __init__ import __author__, __version__

from matplotlib import pyplot as plt
import numpy as np


class LinearRegression:
	X = np.array([])
	Y = np.array([])
	W = np.array([])

	mode = 'None'

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

	def train(self, mode, **kwargs):
		if mode not in LinearRegression.modes.keys():
			raise NameError("Invalid training mode! Must be either 'gradient' or 'normal'.")

		self.mode = mode
		return LinearRegression.modes[mode](self, kwargs)

	def __train_gradient__(self, kwargs):
		alpha = 0.1
		threshold = 0.1
		if 'alpha' in kwargs.keys():
			alpha = kwargs['alpha']
		if 'threshold' in kwargs.keys():
			threshold = kwargs['threshold']

		X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
		self.W = np.ones(self.X.shape[1] + 1).astype(np.float32)
		current_error = threshold * 2.0
		last_error = threshold * 3.0
		N = self.X.shape[0]
		_W = np.array(self.W)

		while np.abs(last_error - current_error) > threshold and not (np.any(np.isnan(_W) == True) and np.any(np.isinf(_W) == True)):
			last_error = current_error
			H = LinearRegression.__predictive_model__(X, _W)
			current_error = LinearRegression.__cost_function__(N, H, self.Y)

			self.W = _W
			_W = np.array(self.W)
			for i, w in enumerate(self.W):
				derivated = LinearRegression.__derived__(i, X, self.W, self.Y)
				_W[i] = w - ((float(alpha) / float(N)) * derivated)
			print('cost:', current_error, 'W:', self.W, 'H:', H)

		return self

	def __train_normal__(self, kwargs):
		X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))

		self.W = np.dot(
			np.linalg.inv(np.dot(X.T, X)),
			np.dot(X.T, self.Y)
		)
		return self

	@staticmethod
	def __cost_function__(N, H, Y):
		return (1.0 / (2.0 * float(N))) * np.sum(np.power(np.subtract(H, Y), 2))

	@staticmethod
	def __predictive_model__(X, W):
		return np.dot(X, W)

	@staticmethod
	def __derived__(i, X, W, Y):
		if i == 0:
			return LinearRegression.__derived_theta_zero__(X, W, Y)
		return LinearRegression.__derived_theta_i__(X, W, Y)

	@staticmethod
	def __derived_theta_zero__(X, W, Y):
		return np.sum(
			np.subtract(
				np.dot(
					X,
					W
				),
				Y
			)
		)

	@staticmethod
	def __derived_theta_i__(X, W, Y):
		return np.sum(
			np.multiply(
				np.subtract(
					np.dot(
						X,
						W
					),
					Y
				),
				X.T[1]
			)
		)

	def test(self, X):
		if self.W.shape[0] == 0:
			raise NameError('Impossible to test while not training!')

		X = np.vstack((np.ones(X.shape[0]), X))
		return np.dot(X.T, self.W)

	def plot(self, model_X, model_Y):
		plt.scatter(self.X, self.Y, label='f(x)')

		if isinstance(model_Y, list) or isinstance(model_Y, np.ndarray):
			plt.plot(model_X, model_Y, label='h(x)')
			plt.legend(title=self.mode + ' mode')
		elif isinstance(model_Y, dict):
			for y in model_Y.items():
				plt.plot(model_X, y[1], label=y[0])
		plt.legend()
		plt.show()

	'''
	pointer to class instance methods must be placed at the end of the class scope
	'''
	modes = {'gradient': __train_gradient__, 'normal': __train_normal__}


def main():
	"""
	Restrictions:
	use for univariate regression only!
	"""
	np.random.seed(1)
	#dataset = np.random.randint(0, 10, 10 * 2).reshape(10, 2)
	dataset = np.array([
		[5, 4],
		[3, 4],
		[0, 1],
		[4, 3]
	])

	model = LinearRegression()
	model.fit(dataset, class_index=-1, skip_header=True)

	x_axis = np.arange(np.min(model.X), np.max(model.X), 0.1)

	model.train('normal')
	normal_y = model.test(x_axis)
	print('normal:', model.W)
	model.train('gradient', alpha=0.1)
	gradient_y = model.test(x_axis)

	model.plot(x_axis, {'normal': normal_y, 'gradient': gradient_y})


if __name__ == '__main__':
	main()
