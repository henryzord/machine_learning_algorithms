__author__ = 'Henry'

import numpy as np
from matplotlib import pyplot as plt


class SingleLinkage:

	distance_matrix = np.array([])
	distance_func = None
	diagonal_value = None
	update_method = None
	label = []
	joined = ''
	y = 1
	x = []
	current_min_value = 0

	def __init__(self, func, update_method, low, high, shape):
		self.distance_func = func
		self.diagonal_value = high + 1
		# distance_matrix = np.random.randint(low, high, shape)
		# self.distance_matrix = SingleLinkage.__clear_diagonal__(distance_matrix, self.diagonal_value)
		self.update_method = update_method

		self.distance_matrix = np.array([
			[11, 11, 11, 11, 11],
		    [ 2, 11, 11, 11, 11],
		    [ 6,  5, 11, 11, 11],
		    [10,  9,  4, 11, 11],
		    [ 9,  8,  5,  3, 11]
		])

		self.label = ['x' + str(i) for i in range(self.distance_matrix.shape[0])]
		self.x = range(self.distance_matrix.shape[0])
		self.y = 1

		for i, label in enumerate(self.label):
			plt.scatter(self.x[i], self.y, marker='o')
			plt.annotate(xy=(self.x[i], self.y), xytext=(self.x[i], self.y), s=label)

	@staticmethod
	def __clear_diagonal__(distance_matrix, value):
		for i in range(distance_matrix.shape[0]):
			for j in range(0, i + 1):
				distance_matrix[i][j] = value
				# if i == j:
				# 	distance_matrix[i][j] = high + 1
				# else:
				# 	distance_matrix[i][j] = distance_matrix[j][i]

		return distance_matrix.T

	def cluster(self):
		while self.distance_matrix.shape[0] > 1:
			min_value = np.min(self.distance_matrix)
			self.current_min_value = min_value
			self.y += min_value

			line = np.where(self.distance_matrix == min_value)[0][0]
			column = np.where(self.distance_matrix[line] == min_value)[0][0]
			coordinates = tuple((line, column))
			self.update_methods[self.update_method](self, coordinates, min_value)

		plt.show()

	def __reduce_matrix__(self, coordinates, min_value):
		column = []

		distance_matrix = self.distance_matrix

		if all([x < (float(self.distance_matrix.shape[0]) / 2.) for x in list(coordinates)]):
			for i in range(self.distance_matrix.shape[0]):
				column.append(self.distance_func((self.distance_matrix[i][coordinates[0]], self.distance_matrix[i][coordinates[1]])))

			column[coordinates[0]] = self.diagonal_value
			column[coordinates[1]] = self.diagonal_value

			distance_matrix[:, min(coordinates)] = column

		else:
			for i in range(self.distance_matrix.shape[0]):
				column.append(self.distance_func((self.distance_matrix[coordinates[0]][i], self.distance_matrix[coordinates[1]][i])))

			column[coordinates[0]] = self.diagonal_value
			column[coordinates[1]] = self.diagonal_value

			distance_matrix[min(coordinates), :] = column

		for item in self.x:
			plt.plot([item, item], [self.y - self.current_min_value, self.y], c='black')

		plt.plot([self.x[coordinates[0]], self.x[coordinates[1]]], [self.y, self.y], c='black')
		plt.annotate(xy=(self.x[coordinates[0]], self.y), xytext=(self.x[coordinates[0]], self.y), s=min_value)
		self.label[min(coordinates)] = '(' + self.label[coordinates[1]] + ',' + self.label[coordinates[0]] + ')'
		self.x[min(coordinates)] = float(self.x[coordinates[0]] + self.x[coordinates[1]]) / 2.
		del self.label[max(coordinates)]
		del self.x[max(coordinates)]

		self.joined = self.label[min(coordinates)]

		distance_matrix = np.delete(distance_matrix, max(coordinates), axis=0)
		distance_matrix = np.delete(distance_matrix, max(coordinates), axis=1)

		self.distance_matrix = distance_matrix

	def __replace_matrix__(self):
		pass

	def __keep_matrix__(self):
		pass

	def build_dendogram(self):
		pass

	update_methods = {'reduce': __reduce_matrix__, 'replace': __replace_matrix__, 'keep': __keep_matrix__}


def main():
	np.random.seed(2)
	algorithm = SingleLinkage(np.max, 'reduce', 1, 10, (5, 5))
	algorithm.cluster()
	print algorithm.joined

main()
