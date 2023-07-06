from __init__ import __author__, __version__

import numpy as np


class MarginalEDA:
    """
    A thug EDA.
    """
    weights = np.array([])
    selection_rate = 0.0
    replacement_rate = 0.0

    def __init__(self, weights, selection_rate, replacement_rate, seed=None):
        """
        Basic Constructor.
        :param weights: Array of weights containing the probability to generate an 1 at every chromosome position (gene).
        :param selection_rate: The rate to select the fittest individuals for any given population.
        :param replacement_rate: The rate to replace older individuals with new ones.
        :param seed: Optional; seed for the numpy.random module.
        """
        np.random.seed(seed)

        self.weights = MarginalEDA.__complement_weights__(weights)

        self.selection_rate = selection_rate
        self.replacement_rate = replacement_rate

    @staticmethod
    def __complement_weights__(weights):
        """
        Given an array with probabilities to generate 1's at each array's position,
        includes the probabilities to generated the zeros alogside the 1's probabilities.
        For example:
        [0.8, 0.3] -> [[0.2, 0.8], [0.7, 0.3]]
        :param weights: The probability array with only the 1's probabilities.
        :return: A matrix containing the probabilities to generate both zeros and ones at each array's positions.
        """
        complement = np.ones(weights.shape[0]) - weights
        return np.vstack((complement, weights)).T

    @staticmethod
    def __sample__(pop_size, weights):
        """
        Generates a sample population based on the probability array.
        :param pop_size: The size of the population to be generated.
        :param weights: Probability array.
        :return: A population denoted by a numpy.ndarray.
        """
        population = np.array([])
        for i in range(0, pop_size):
            population = np.hstack((population, map(lambda x: np.random.choice(2, p=x), weights)))

        population = population.reshape(pop_size, weights.shape[0])
        return population

    @staticmethod
    def __evaluate__(fitness_func, population):
        """
        Evaluates a given population based on a given fitness function.
        :param fitness_func: A pointer to a fitness function.
        :param population: The population to be evaluated.
        :return: An array containing the fitness for each individual in the population.
        """
        return np.array(map(fitness_func, population))

    @staticmethod
    def __max_convergence__(population):
        return int(np.sum(population)) == population.itemsize

    def converge(self, pop_size, fitness_function, iterations=np.inf):
        """
        Explores the solution space seeking for the best probability array.
        :param pop_size: The population size
        :param fitness_function: A pointer to a fitness function, defined outside the scope of this class.
        :param iterations: Number of max iterations
        :return: None -- modifies this instance's probability array.
        """
        population = MarginalEDA.__sample__(pop_size, self.weights)

        iteration = 0
        while not self.__max_convergence__(population) and iteration < iterations:
            fittest = np.array(
                sorted(zip(population, self.__evaluate__(fitness_function, population)), key=lambda x: x[1],
                       reverse=True))[:round(self.selection_rate * pop_size), 0]
            fittest = np.array(map(np.array, fittest)).astype(np.float32)
            fittest_marginal_frequency = np.sum(fittest, axis=0) / float(fittest.shape[0])
            population_marginal_frequency = np.sum(population, axis=0) / float(population.shape[0])

            marginal_frequency = population_marginal_frequency + (
                self.replacement_rate * (fittest_marginal_frequency - population_marginal_frequency))
            self.weights = MarginalEDA.__complement_weights__(marginal_frequency)
            # print self.weights[:, 1]
            population = MarginalEDA.__sample__(pop_size, self.weights)
            iteration += 1

        print('data at', iteration, 'iterations:')
        print('probability array:', self.weights[:, 1])


def fitness_function(individual):
    return np.sum(individual)


def main():
    weights = np.tile(np.array([0.5]), 100)
    pop_size = 25  # apperently this is the minimum population size for the EDA to converge with the below parameters
    selection_rate = 0.6
    replacement_rate = 0.4

    my_eda = MarginalEDA(weights, selection_rate, replacement_rate)
    my_eda.converge(pop_size, fitness_function, 100)


if __name__ == '__main__':
    main()

