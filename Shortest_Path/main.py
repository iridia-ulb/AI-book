from AlgorithmEuclidean import AlgorithmEuclidean
from AlgorithmChebyshev import AlgorithmChebyshev
from AlgorithmMean import AlgorithmMean
from Dijkstra import Dijkstra


if __name__ == "__main__":

    input_file = "datasets/20_nodes.txt"

    algorithm = AlgorithmEuclidean(input_file, False)

    algorithm = AlgorithmChebyshev(input_file, False)

    algorithm = AlgorithmMean(input_file, False)

    dijkstra = Dijkstra(input_file)

