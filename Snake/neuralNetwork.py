import numpy as np
import copy as cp
import random as rd


def addBias(val):
    """
    Add one element to the column vector to take into account the bias.
    val (list): the column vector
    """
    return np.column_stack((val, np.matrix([[1]])))


class NeuralNetwork:
    """
    Neural network that uses the DNA (weigts and biases) of a snake to predict its movement.
    """

    def __init__(self, weights, biases):
        """
        Constructor.
        weights (list): weights used by the neural network
        bias (list): bias used by the neural network
        """
        self.weights = cp.deepcopy(weights)

        for layer in range(len(biases)):
            self.weights[layer].append(cp.deepcopy(biases[layer]))

        for i in range(len(self.weights)):
            self.weights[i] = np.matrix(self.weights[i])

    def predict(self, inputs):
        """
        Predict the next movement of the snake.
        inputs (list): The vision of the snake
        outputs (list): Outputs of the neural network (one for each direction)
        """
        outputs = np.matrix([inputs])

        for layerWeights in self.weights:
            outputs = addBias(outputs)
            outputs = outputs.dot(layerWeights)
            outputs[outputs < 0] = 0  # Relu function

        return outputs
