from sudoku_alg import valid
import sys
import time
import numpy as np
import pygame
from random import choice


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
            if board[i][j] == 0:
                # Cell is filled with a number, don't need to check the possibilities.
                possibilities[i][j] = np.array(
                    [p if valid(board, (i, j), p) else -1 for p in range(1, 10)]
                )
    return possibilities


def getCellToExplore(heuristics):
    """
    Returns the best cell to explore according to the heuristics matrix.
    np.amin : Takes all the minimum elements in the heuristic array, where the values are > 0.
    np.where : Takes the coordinates (i, j) of all the minimum elements.
    np.dstack : Unzip the result of np.where to have the coordinates in the good format.
    Finally get a random cell from all the best cells.
    """
    cell = choice(
        np.dstack(np.where(heuristics == np.amin(heuristics[heuristics > 0])))[0]
    )
    return cell


def updateHeuristics(board, heuristics, possibilities):
    """
    Updates the heuristics matrix
    mode: the technique used to determine the heuristic
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                heuristics[i][j] = (
                    len(possibilities[i][j][possibilities[i][j] >= 1]) + 1
                )
            else:
                heuristics[i][j] = 0
    return heuristics


def aStarBacktracking(board, possibilities, heuristics):
    """
    Fills the solution matrix by backtracking in the order given by the heuristics matrix.
    Implements the A* algorithm on the base of this heuristics matrix.
    """
    if not isFull(board):
        # If there are still cells not filled
        possibilities = reducePossibilities(board, possibilities)
        # Adapts the possible numbers for each cell
        heuristics = updateHeuristics(board, heuristics, possibilities)
        cell = getCellToExplore(heuristics)
        # This gets the cell with the best heuristic
        for number in possibilities[cell[0]][cell[1]][
            possibilities[cell[0]][cell[1]] > 0
        ]:
            # For each possible solution to fill the cell
            board[cell[0]][cell[1]] = number
        board[cell[0]][cell[1]] = 0
        # If using this number led to a wrong state, we reset it to zero
        return False
    return True


class AstarSolver:
    def __init__(self, game, startTime):
        self.game = game
        self.heuristics = np.array([[0 for _ in range(9)] for _ in range(9)])
        self.possibilities = np.array(
            [[list(range(1, 10)) for _ in range(9)] for _ in range(9)]
        )
        self.startTime = startTime

    def visualSolve(self, wrong):
        """
        Solve the board with A*
        """

        if not isFull(self.game.board):
            possibilities = reducePossibilities(self.game.board, self.possibilities)
            # Adapts the possible numbers for each cell
            self.heuristics = updateHeuristics(
                self.game.board, self.heuristics, possibilities
            )
            cell = getCellToExplore(self.heuristics)
            # This gets the cell with the best heuristic
            for number in possibilities[cell[0]][cell[1]][
                possibilities[cell[0]][cell[1]] >= 1
            ]:
                for event in pygame.event.get():
                    # so that touching anything doesn't freeze the screen
                    if event.type == pygame.QUIT:
                        sys.exit()
                self.game.tries += 1
                self.game.board[cell[0]][cell[1]] = number
                self.game.tiles[cell[0]][cell[1]].value = number
                self.game.tiles[cell[0]][cell[1]].correct = True

                # pygame.time.delay(63) #show tiles at a slower rate

                self.game.redraw({}, wrong, time.time() - self.startTime)
                if self.visualSolve(wrong):
                    # Then, we try to solve the sudoku starting from the updated board
                    return True
            self.game.board[cell[0]][cell[1]] = 0
            self.game.tiles[cell[0]][cell[1]].value = 0
            self.game.tiles[cell[0]][cell[1]].incorrect = True
            self.game.tiles[cell[0]][cell[1]].correct = False
            # pygame.time.delay(63)
            self.game.redraw({}, wrong, time.time() - self.startTime)
            return False
        return True
