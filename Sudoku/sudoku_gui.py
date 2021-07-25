from sudoku_alg import valid, solve, find_empty
from copy import deepcopy
import sys
import pygame
import time
import random
import argparse
from pathlib import Path
import numpy as np
from astar import AstarSolver
from genetic_algorithm import GeneticSolver


def readSudokus(filename):
    """
    Returns the n first sudokus of the file with given name
    """
    f = open(filename)
    res = None
    txt = f.readline().strip()
    if txt != "":
        res = [[int(txt[i + j * 9]) for i in range(9)] for j in range(9)]
    f.close()
    return np.array(res)


class Board:
    """A sudoku board made out of Tiles"""

    def __init__(self, window, file=Path("sudokus") / Path("sudoku1.txt")):
        self.board = readSudokus(file)
        self.initialBoard = deepcopy(self.board)
        self.solvedBoard = deepcopy(self.board)
        solve(self.solvedBoard)
        self.tiles = np.array(
            [
                [Tile(self.board[i][j], window, i * 60, j * 60) for j in range(9)]
                for i in range(9)
            ]
        )
        self.window = window
        self.tries = 0

    def draw_board(self):
        """Fills the board with Tiles and renders their values"""
        for i in range(9):
            for j in range(9):
                if j % 3 == 0 and j != 0:  # vertical lines
                    pygame.draw.line(
                        self.window,
                        (0, 0, 0),
                        ((j // 3) * 180, 0),
                        ((j // 3) * 180, 540),
                        4,
                    )

                if i % 3 == 0 and i != 0:  # horizontal lines
                    pygame.draw.line(
                        self.window,
                        (0, 0, 0),
                        (0, (i // 3) * 180),
                        (540, (i // 3) * 180),
                        4,
                    )

                self.tiles[i][j].draw((0, 0, 0), 1)

                if self.tiles[i][j].value != 0:  # don't draw 0s on the grid
                    self.tiles[i][j].display(
                        self.tiles[i][j].value,
                        (21 + (j * 60), (16 + (i * 60))),
                        (0, 0, 0),
                    )  # 20,5 are the coordinates of the first tile
        # bottom-most line
        pygame.draw.line(
            self.window,
            (0, 0, 0),
            (0, ((i + 1) // 3) * 180),
            (540, ((i + 1) // 3) * 180),
            4,
        )

    def deselect(self, tile):
        """Deselects every tile except the one currently clicked"""
        for i in range(9):
            for j in range(9):
                if self.tiles[i][j] != tile:
                    self.tiles[i][j].selected = False

    def redraw(self, keys, wrong, elapsed, gen=None):
        """Redraws board with highlighted tiles"""
        self.window.fill((255, 255, 255))
        self.draw_board()
        for i in range(9):
            for j in range(9):
                if self.tiles[j][i].selected:  # draws the border on selected tiles
                    self.tiles[j][i].draw((50, 205, 50), 4)

                elif self.tiles[i][j].correct:
                    self.tiles[j][i].draw((34, 139, 34), 4)

                elif self.tiles[i][j].incorrect:
                    self.tiles[j][i].draw((255, 0, 0), 4)

        if len(keys) != 0:
            # draws inputs that the user places on board but not their final value on that tile
            for value in keys:
                self.tiles[value[0]][value[1]].display(
                    keys[value],
                    (21 + (value[0] * 60), (16 + (value[1] * 60))),
                    (128, 128, 128),
                )

        if wrong > 0:
            font = pygame.font.SysFont("Bauhaus 93", 30)  # Red X
            text = font.render("X", True, (255, 0, 0))
            self.window.blit(text, (10, 554))

            # Number of Incorrect Inputs
            font = pygame.font.SysFont("Bahnschrift", 40)
            text = font.render(str(wrong), True, (0, 0, 0))
            self.window.blit(text, (32, 542))

        font = pygame.font.SysFont("Bahnschrift", 40)  # Time Display
        passedTime = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        text = font.render(str(passedTime), True, (0, 0, 0))
        triesText = font.render(str(self.tries), True, (0, 0, 0))
        self.window.blit(text, (388, 542))
        if gen is not None:
            font = pygame.font.SysFont("Bahnschrift", 40)  # Time Display
            text = font.render("Generation: " + str(gen), True, (50, 50, 50))
            self.window.blit(text, (38, 542))
        else:
            self.window.blit(triesText, (300, 542))
        pygame.display.flip()

    def backVisualSolve(self, wrong, time):
        """Showcases how the board is solved via backtracking"""
        for event in pygame.event.get():
            # so that touching anything doesn't freeze the screen
            if event.type == pygame.QUIT:
                sys.exit()

        empty = find_empty(self.board)
        if not empty:
            return True

        for nums in range(9):
            if valid(self.board, (empty[0], empty[1]), nums + 1):
                self.board[empty[0]][empty[1]] = nums + 1
                self.tiles[empty[0]][empty[1]].value = nums + 1
                self.tiles[empty[0]][empty[1]].correct = True
                pygame.time.delay(63)  # show tiles at a slower rate
                self.redraw({}, wrong, time)

                if self.backVisualSolve(wrong, time):
                    return True

                self.board[empty[0]][empty[1]] = 0
                self.tiles[empty[0]][empty[1]].value = 0
                self.tiles[empty[0]][empty[1]].incorrect = True
                self.tiles[empty[0]][empty[1]].correct = False
                pygame.time.delay(63)
                self.redraw({}, wrong, time)

    def hint(self, keys):
        """Shows a random empty tile's solved value as a hint"""
        while True:  # keeps generating i,j coords until it finds a valid random spot
            i = random.randint(0, 8)
            j = random.randint(0, 8)
            if self.board[i][j] == 0:  # hint spot has to be empty
                if (j, i) in keys:
                    del keys[(j, i)]
                self.board[i][j] = self.solvedBoard[i][j]
                self.tiles[i][j].value = self.solvedBoard[i][j]
                return True

            elif self.board == self.solvedBoard:
                return False


class Tile:
    """Represents each white tile/box on the grid"""

    def __init__(self, value, window, x1, y1):
        self.value = value
        self.window = window
        self.rect = pygame.Rect(x1, y1, 60, 60)  # dimensions for the rectangle
        self.selected = False
        self.correct = False
        self.incorrect = False

    def draw(self, color, thickness):
        """Draws a tile on the board"""
        pygame.draw.rect(self.window, color, self.rect, thickness)

    def display(self, value, position, color):
        """Displays a number on that tile"""
        font = pygame.font.SysFont("lato", 45)
        text = font.render(str(value), True, color)
        self.window.blit(text, position)

    def clicked(self, mousePos):
        """Checks if a tile has been clicked"""
        if self.rect.collidepoint(mousePos):  # checks if a point is inside a rect
            self.selected = True
        return self.selected


def main():
    """Runs the main Sudoku GUI/Game"""
    parser = argparse.ArgumentParser(description="Launch the sudoku game")
    parser.add_argument("-a", "--algorithm", help="Choose the algorithm to execute")
    parser.add_argument("-f", "--file", help="Sudoky instance to solve")
    args = parser.parse_args()

    screen = pygame.display.set_mode((540, 590))
    screen.fill((255, 255, 255))
    pygame.display.set_caption("Sudoku")

    # loading screen when generating grid
    font = pygame.font.SysFont("Bahnschrift", 40)
    text = font.render("Generating", True, (0, 0, 0))
    screen.blit(text, (175, 245))

    font = pygame.font.SysFont("Bahnschrift", 40)
    text = font.render("Grid", True, (0, 0, 0))
    screen.blit(text, (230, 290))
    pygame.display.flip()

    # initiliaze values and variables
    startTime = time.time()
    wrong = 0
    if Path(args.file).is_file():
        board = Board(screen, args.file)
    else:
        print("No sudoku selected, using default one: sudokus/sudoku1.txt")
        board = Board(screen)

    if args.algorithm == "astar":
        solver = AstarSolver(board, startTime)
    elif args.algorithm == "genetic":
        solver = GeneticSolver(board, startTime)
    else:
        print("Please choose an algorithm")
        sys.exit()

    selected = (
        -1,
        -1,
    )
    # NoneType error when selected = None, easier to just format as a tuple whose value will never be used
    keyDict = {}
    running = True
    while running:
        elapsed = time.time() - startTime

        if board.board.all() == board.solvedBoard.all():  # user has solved the board
            for i in range(9):
                for j in range(9):
                    board.tiles[i][j].selected = False
                    running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()  # so that it doesnt go to the outer run loop

            elif event.type == pygame.MOUSEBUTTONUP:
                # allow clicks only while the board hasn't been solved
                mousePos = pygame.mouse.get_pos()
                for i in range(9):
                    for j in range(9):
                        if board.tiles[i][j].clicked(mousePos):
                            selected = i, j
                            # deselects every tile except the one currently clicked
                            board.deselect(board.tiles[i][j])

            elif event.type == pygame.KEYDOWN:
                if board.board[selected[1]][selected[0]] == 0 and selected != (-1, -1):
                    if event.key == pygame.K_1:
                        keyDict[selected] = 1

                    if event.key == pygame.K_2:
                        keyDict[selected] = 2

                    if event.key == pygame.K_3:
                        keyDict[selected] = 3

                    if event.key == pygame.K_4:
                        keyDict[selected] = 4

                    if event.key == pygame.K_5:
                        keyDict[selected] = 5

                    if event.key == pygame.K_6:
                        keyDict[selected] = 6

                    if event.key == pygame.K_7:
                        keyDict[selected] = 7

                    if event.key == pygame.K_8:
                        keyDict[selected] = 8

                    if event.key == pygame.K_9:
                        keyDict[selected] = 9

                    elif (
                        event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE
                    ):
                        # clears tile out
                        if selected in keyDict:
                            board.tiles[selected[1]][selected[0]].value = 0
                            del keyDict[selected]

                    elif event.key == pygame.K_RETURN:
                        if selected in keyDict:
                            # clear tile when incorrect value is inputted
                            if (
                                keyDict[selected]
                                != board.solvedBoard[selected[1]][selected[0]]
                            ):
                                # clear tile when incorrect value is inputted
                                wrong += 1
                                board.tiles[selected[1]][selected[0]].value = 0
                                del keyDict[selected]
                                break
                            # valid and correct entry into cell
                            # assigns current grid value
                            board.tiles[selected[1]][selected[0]].value = keyDict[
                                selected
                            ]
                            # assigns to actual board so that the correct value can't be modified
                            board.board[selected[1]][selected[0]] = keyDict[selected]
                            del keyDict[selected]

                if event.key == pygame.K_h:
                    board.hint(keyDict)

                if event.key == pygame.K_SPACE:
                    for i in range(9):
                        for j in range(9):
                            board.tiles[i][j].selected = False
                    keyDict = {}  # clear keyDict out

                    solver.visualSolve(wrong)

                    for i in range(9):
                        for j in range(9):
                            board.tiles[i][j].correct = False
                            board.tiles[i][j].incorrect = False  # reset tiles
                    running = False

        board.redraw(keyDict, wrong, elapsed)
    while True:
        # another running loop so that the program ONLY closes when user closes program
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
