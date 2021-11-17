import random as rd
import copy as cp
import numpy as np


class Dna:
    """
    Represents the DNA of a snake by using a list of weights and a list of biases.
    It is possible to modify the DNA by mutating it or by performing a crossover.
    """

    def __init__(self, weights=None, biases=None, layersSize=None):
        """
        Constructor.
        weights (list): A list of weights
        biases (list): A list of biases
        layersSize (list): A list containing the number of hidden neurons used by the neural network
        """
        self.weights = cp.deepcopy(weights)
        self.bias = cp.deepcopy(biases)

        if self.weights is None:
            self.initialize_rd_weights(cp.copy(layersSize))
            self.initialize_rd_bias()
        elif self.bias is None:
            self.initialize_rd_bias()

    def initialize_rd_weights(self, layersSize=None):
        """
        Initializes randomly the weights by using a gaussian distribution of mean 0 and standard deviation of 0.5.
        layersSize (list): A list containing the number of hidden neurons used by the neural network
        """
        self.weights = []

        if layersSize is None:
            layersSize = [rd.randint(10, 15) for i in range(rd.randint(1, 3))]

        layersSize.append(4)  # Add the size of the output layer
        prevNbrNode = 26  # Number of input nodes

        for nextNbrNode in layersSize:  # For each layer
            layer = []
            for j in range(prevNbrNode):  # For each node
                node = [rd.gauss(0, 0.5) for i in range(nextNbrNode)]
                layer.append(node)
            self.weights.append(np.matrix(layer))
            prevNbrNode = nextNbrNode

    def initialize_rd_bias(self):
        """
        Initializes randomly the biases by using a gaussian distribution of mean 0 and standard deviation of 0.5.
        """
        self.bias = []

        for layer in self.weights:
            nbrBias = np.size(layer, axis=1)
            self.bias.append(
                np.array([rd.gauss(0, 0.5) for i in range(nbrBias)])
            )

    def predict(self, inputs):
        """
        Predict the next movement of the snake.
        inputs (list): The vision of the snake
        outputs (list): Outputs of the neural network (one for each direction)
        """
        weights = []
        for layer in range(len(self.bias)):
            weights.append(np.vstack((self.weights[layer], self.bias[layer])))

        outputs = np.matrix([inputs])

        for layerWeights in weights:
            outputs = self.addBias(outputs)
            outputs = outputs.dot(layerWeights)
            outputs[outputs < 0] = 0  # Relu function

        return outputs

    def addBias(self, val):
        """
        Add one element to the column vector to take into account the bias.
        val (list): the column vector
        """
        return np.column_stack((val, np.matrix([[1]])))

    def mix(self, other, mutationRate=0.01):
        """
        Mix the copy of this DNA with the copy of another one to create a new one.
        A crossover is first performed on the two DNA, giving a new one, then the new DNA is mutated.
        other (Dna): The other DNA used for the mixing
        mutationRate (float): The probability for a weight or bias to be mutated
        """
        newWeights = self.crossover(self.weights, other.weights)
        newBias = self.crossover(self.bias, other.bias)
        newDna = Dna(newWeights, newBias)
        newDna.mutate(mutationRate)
        return newDna

    def cross_layer(self, layer1, layer2):
        """
        Performs a crossover on two layers.
        layer1 (list): The first layer used to do the crossover
        layer2 (list): The second layer used to do the crossover
        Returns a copy of the result (list)
        """
        lineCut = rd.randint(0, np.size(layer1, axis=0) - 1)
        if len(layer1.shape) == 1:  # the layer is only one dimension
            return np.hstack((layer1[:lineCut], layer2[lineCut:]))

        columnCut = rd.randint(0, np.size(layer1, axis=1) - 1)
        res = np.vstack(
            (
                layer1[:lineCut],
                np.hstack(
                    (layer1[lineCut, :columnCut], layer2[lineCut, columnCut:])
                ),
                layer2[lineCut + 1 :],
            )
        )
        return res

    def crossover(self, dna1, dna2):
        """
        Performs a crosover on the layers (weights and biases).
        dna1 (Dna): The first DNA on which the crossover is performed
        dna2 (Dna): The second DNA on which the crossover is performed
        Returns the crossover of the two DNA (list)
        """
        res = []

        for layer in range(len(dna1)):
            newLayer = self.cross_layer(dna1[layer], dna2[layer])
            res.append(newLayer)

        return res

    def mutate_layer(self, layer, mutationRate=0.01):
        """
        Mutate a layer by adding a value from a gaussian distribution of mean 0 and standard deviation of 0.5
        layer (list): the layer that is mutated
        mutationRate (float): The probability for a value of the layer to be mutated
        """
        with np.nditer(layer, op_flags=["readwrite"]) as it:
            for x in it:
                if rd.random() < mutationRate:
                    x[...] += min(max(rd.gauss(0, 0.5), -1), 1)

    def mutate(self, mutationRate=0.01):
        """
        Mutatate the DNA.
        mutationRate (float): The probability for a value of the layer to be mutated
        """
        # Mutation of the weights
        for layer in self.weights:
            self.mutate_layer(layer, mutationRate)

        # Mutation of the bias
        for layer in self.bias:
            self.mutate_layer(layer, mutationRate)

    def __str__(self):
        """
        Returns the DNA's information as a string
        """
        return f"weights: {self.weights}\n\nbias: {self.bias}"
