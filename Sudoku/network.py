import numpy as np
import matplotlib.pyplot as plt

def createMeshGrid(arr):
    n = Network()
    n.cells = arr
    ax = [[0 for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            t = plt.subplot(9, 9, i*9+j+1)
            t.set_axis_off()
            t.pcolormesh(n.getMatrixAt(i, j), )   
    plt.show()

class Network:
    """
    This class represent the network underlying an average sudoku grid.
     - self.cells[a][b][c][d] contains the intensity of heuristic change for the cell (c, d) when the cell (a, b) is filled
     - self.attempts is a counter on the number of guesses
     - self.heuristic contains the heuristic associated to the currently filled sudoku
    """
    def __init__(self):
        self.cells = np.array([[np.array([[0 for _ in range(9)] for _ in range(9)]) for _ in range(9)] for _ in range(9)])
        self.attempts = np.array([[0 for _ in range(9)] for _ in range(9)])
        self.heuristic = np.array([[0 for _ in range(9)] for _ in range(9)])
    
    def getMatrixAt(self, a, b):
        """
        Returns the graph linked to the cell with coordinates (a, b)
        """
        return self.cells[a][b]
    
    def prettyPrint(self):
        """
        Displays the averaged weights of the Network, with aim to be piped in a python file, so that
        the Network can be used in the 3rd method of heuristics in Astar
        """
        print("import numpy as np\nfrom network import *\nweights = np.array([", end="")
        for i in range(9):
            print("[", end="")
            for j in range(9):
                print('[', end="")
                for k in range(9):
                    if k!=8:
                        print([self.cells[i][j][k][m]/max(self.attempts[i][j], 1) for m in range(9)], end=",")
                    else:
                        print([self.cells[i][j][k][m]/max(self.attempts[i][j], 1) for m in range(9)], end="")
                if j!=8:
                    print("],", end="")
                else:
                    print("]", end="")
            if i!=8:
                print("],", end="")
            else:
                print("]])")
        print("""if __name__=="__main__":\n\tcreateMeshGrid(weights)""")
                   
    def updateHeuristics(self, difference, a, b):
        """
        Updates the cells matrix for the cell (a,b)
        """
        difference[difference < 0] = 0
        difference[a][b] = 0
        self.cells[a][b] += difference
        self.attempts[a][b] += 1
    
    def average(self):
        """
        Averages the weights depending on the number of attempts
        """
        np.divide(self.cells, self.attemps) 