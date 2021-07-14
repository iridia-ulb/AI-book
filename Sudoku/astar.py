import sys

from sudoku_alg import *
from network import * 
from readSudokus import *
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from Weights31 import weights

tries = 0
network = Network()


def isFull(board):
    """
    Check if all the cells in the board are filled (the sudoku is completed).
    """
    res = True
    for i in board:
        res = res and not (0 in i)
    return res

def reducePossibilities(board, possibilities):
    """
    Reduces the possibilities for each cell.
    If a number is not possible, put a "-1" instead.
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:    # Cell is filled with a number, don't need to check the possibilities.
                possibilities[i][j] = np.array([p if valid(board, (i, j), p) else -1 for p in range(1, 10)])
    return possibilities

def getCellToExplore(heuristics):
    """
    Returns the best cell to explore according to the heuristics matrix.
    np.amin : Takes all the minimum elements in the heuristic array, where the values are > 0.
    np.where : Takes the coordinates (i, j) of all the minimum elements.
    np.dstack : Unzip the result of np.where to have the coordinates in the good format.
    Finally get a random cell from all the best cells.
    """
    cell = choice(np.dstack(np.where(heuristics == np.amin(heuristics[heuristics > 0])))[0])
    return cell

def computeSmallCrossedHeuristics(i, j, possibilities, minimal):
    """
    Returns the total number of possibilities that the cell (i,j) is linked to (row, column, square).
    """
    total = len(possibilities[i][j][possibilities[i][j] >= 1]) + 1   # Possibilities of the cell.
    if total > 3 and total <= minimal:  # If total < 3, interesting possibilities that we do not want to lose. 
        total -= 1
        for k in range(9):
            if total <= minimal and k != j:
                total += len(possibilities[i][k][possibilities[i][k] >= 1])   # Row
        for k in range(9):
            if total <= minimal and k != i:
                total += len(possibilities[k][j][possibilities[k][j] >= 1])   # Columns
        start_row = i - i % 3       # Get the first row of the square linked to (i,j).
        start_column = j - j % 3    # Get the first column of the square linked to (i,j).
        for k in range(3):  # Square
            for l in range(3):
                if total <= minimal and (k != i or l != j):
                    total += len(possibilities[start_row + k][start_column + l][possibilities[start_row + k][start_column + l] >= 1])
    return total


def computeBigCrossedHeuristics(i, j, possibilities, minimal):
    """
    Returns the total number of possibilities that the cell (i,j) is linked to (row, column, square).
    And also the "big squares" linked to the rows and columns too.
    """
    total = len(possibilities[i][j][possibilities[i][j] >= 1])+1      # Possibilities of the cell.
    if total > 3 and total <= minimal:  # If total < 3, interesting possibilities that we do not want to lose.
        total = 0
        start_row = i - i % 3       # Get the first row of the square linked to (i,j).
        start_column = j - j % 3    # Get the first column of the square linked to (i,j).
        for k in range(3):
            for l in range(9):
                if total <= minimal:
                    if start_row+k == i:
                        total += len(possibilities[start_row+k][l][possibilities[start_row+k][l] >= 1])   # Rows
                    total += len(possibilities[start_row+k][l][possibilities[start_row+k][l] >= 1])   # Rows
        for k in range(9):
            for l in range(3):
                if total <= minimal:
                    if start_column+l == j:
                        total += len(possibilities[k][start_column+l][possibilities[k][start_column+l] >= 1])   # Columns
                    total += len(possibilities[k][start_column+l][possibilities[k][start_column+l] >= 1])   # Columns
    return total


def computeWeightedHeuristics(board, i, j, possibilities, minimal):
    """
    Returns the total number of possibilities that the cell (i,j) is linked to weighted by the learnt weights.
    """
    total = len(possibilities[i][j][possibilities[i][j] >= 1])+1
    if total > 3 and total <= minimal: # If total < 3, interesting possibilities that we do not want to lose.
        for m in range(9):
            for n in range(9):
                if total <= minimal and board[m][n] == 0:
                    total += len(possibilities[m][n][possibilities[m][n] >= 1])/(max(weights[i][j][m][n], 1))
    return total


def updateHeuristics(board, heuristics, possibilities, mode=0, learning=0):
    """
    Updates the heuristics matrix
    mode: the technique used to determine the heuristic
    """
    previousHeuristic = deepcopy(heuristics)
    minimal = 10000000000 # An enough high value
    if mode == 0:   # Basic heuristic (heuristic = number of possibilities of the cell).
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    heuristics[i][j] = len(possibilities[i][j][possibilities[i][j] >= 1]) + 1
                else:
                    heuristics[i][j] = 0
    elif mode == 1: # Small cross heuristic
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    heuristics[i][j] = computeSmallCrossedHeuristics(i, j, possibilities, minimal)
                    if not learning and heuristics[i][j] < minimal:
                        minimal = heuristics[i][j]
                else:
                    heuristics[i][j] = 0
    elif mode == 2: # Big cross heuristic
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    heuristics[i][j] = computeBigCrossedHeuristics(i, j, possibilities, minimal)
                    if heuristics[i][j] < minimal:
                        minimal = heuristics[i][j]
                else:
                    heuristics[i][j] = 0
    elif mode == 3: # Weighted heuristic
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    heuristics[i][j] = computeWeightedHeuristics(board, i, j, possibilities, minimal)
                    if not learning and heuristics[i][j] < minimal:
                        minimal = heuristics[i][j]
                else:
                    heuristics[i][j] = 0
    if learning == 1: # If we learn, we have to compute the variation of heuristic
        difference = previousHeuristic - heuristics
        return heuristics, difference
    return heuristics


def aStarBacktracking(board, possibilities, heuristics, mode=3, learning=0, a=-1, b=-1):
    """
    Fills the solution matrix by backtracking in the order given by the heuristics matrix.
    Implements the A* algorithm on the base of this heuristics matrix.
    """
    global tries, network
    if not isFull(board):                               # If there are still cells not filled
        possibilities = reducePossibilities(board, possibilities)   # Adapts the possible numbers for each cell
        if learning == 1:
            heuristics, difference = updateHeuristics(board, heuristics, possibilities, mode, learning)
        else:
            heuristics = updateHeuristics(board, heuristics, possibilities, mode, learning)      
        cell = getCellToExplore(heuristics)                         # This gets the cell with the best heuristic
        for number in possibilities[cell[0]][cell[1]][possibilities[cell[0]][cell[1]] > 0]:              # For each possible solution to fill the cell
            board[cell[0]][cell[1]] = number
            tries += 1
            if aStarBacktracking(board, possibilities, heuristics, mode, learning, cell[0], cell[1]):          # Then, we try to solve the sudoku starting from the updated board
                if learning == 1 and a != -1: # if a==-1, then we are at the first try and we don't have any variation in heuristic already computed
                    network.updateHeuristics(difference, a, b)
                return True
        board[cell[0]][cell[1]] = 0                    # If using this number led to a wrong state, we reset it to zero
        return False
    return True


if __name__ == "__main__":
    number_of_sudokus = 20
    all_grids = readSudokus(number_of_sudokus)
    total_tries = 0
    mode = 0
    learning = 0
    for i in range(len(all_grids)):
        print("Solving Sudoku", i, "in ", end="")
        tries = 0
        possibilities = np.array([[list(range(1, 10)) for _ in range(9)] for _ in range(9)])    # This contains the possibilities of number that each cell could contain
        heuristics = np.array([[0 for _ in range(9)] for _ in range(9)])              # This contains the heuristic associated to each case, meaning the interest to fill in order to get the closest to the solution, the smaller the score is, the more likely it is to fill the cell
        aStarBacktracking(deepcopy(all_grids[i]), possibilities, heuristics, mode, learning)  # Starts the filling process
        print(tries, "tries")
        total_tries += tries
    print("Average number of tries:", total_tries/number_of_sudokus)
