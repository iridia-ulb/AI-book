import random as rd
from neuralNetwork import *


def flatten(lst):
    """
    Transform the list lst into a list of one dimension.
    lst (list): the list to flatten
    Returns the flattened list (list).
    """
    if type(lst[0]) != list:
        return lst

    res = []

    for i in lst:
        toAdd = flatten(i)
        res += toAdd

    return res


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
        prevNbrNode = 24  # Number of input nodes

        for nextNbrNode in layersSize:  # For each layer
            layer = []
            for j in range(prevNbrNode):  # For each node
                node = [rd.gauss(0, 0.5) for i in range(nextNbrNode)]
                layer.append(node)
            self.weights.append(layer)
            prevNbrNode = nextNbrNode

    def initialize_rd_bias(self):
        """
        Initializes randomly the biases by using a gaussian distribution of mean 0 and standard deviation of 0.5.
        """
        self.bias = []

        for layer in self.weights:
            nbrBias = len(layer[0])
            self.bias.append([rd.gauss(0, 0.5) for i in range(nbrBias)])

    def get_weights(self):
        """
        Returns the weights (list)
        """
        return self.weights

    def get_bias(self):
        """
        Returns the biases (list)
        """
        return self.bias

    def get_model(self):
        """
        Returns a neural network that uses the same weights and biases (NeuralNetwork).
        """
        return NeuralNetwork(self.weights, self.bias)

    def mix(self, other, mutationRate=0.01):
        """
        Mix the copy of this DNA with the copy of another one to create a new one.
        A crossover is first performed on the two DNA, giving a new one, then the new DNA is mutated.
        other (Dna): The other DNA used for the mixing
        mutationRate (float): The probability for a weight or bias to be mutated
        """
        newWeights = self.crossover(self.weights, other.weights)
        newBias = self.cross_layer(self.bias, other.bias)
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
        # Choose randomly where the crossover is performed
        lineCut = rd.randint(0, len(layer1) - 1)
        columnCut = rd.randint(0, len(layer1[0]) - 1)
        res = layer1[:lineCut]
        res.append(layer1[lineCut][:columnCut] + layer2[lineCut][columnCut:])

        if lineCut < len(layer1) - 1:
            res += layer2[lineCut + 1 :]

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
        for i in range(len(layer)):
            for j in range(len(layer[i])):
                if rd.random() < mutationRate:
                    layer[i][j] += rd.gauss(0, 0.5)  # Maybe need change
                    # Bound the weights if they exceed the limit
                    if layer[i][j] > 1:
                        layer[i][j] = 1
                    elif layer[i][j] < -1:
                        layer[i][j] = -1

    def mutate(self, mutationRate=0.01):
        """
        Mutatate the DNA.
        mutationRate (float): The probability for a value of the layer to be mutated
        """
        # Mutation of the weights
        for layer in self.weights:
            self.mutate_layer(layer, mutationRate)

        # Mutation of the bias
        self.mutate_layer(self.bias, mutationRate)

    def __str__(self):
        """
        Returns the DNA's information as a string
        """
        return f"weights: {self.weights}\n\nbias: {self.bias}"


if __name__ == "__main__":
    print("start")
    NbrNodes = [rd.randint(10, 15) for i in range(rd.randint(1, 3))]
    dna1 = Dna(layersSize=NbrNodes)
    dna2 = Dna(layersSize=NbrNodes)
    print("dna1", dna1)
    print("dna2", dna2)
    print("mix", dna1.mix(dna2))


#
