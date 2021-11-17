import os
import pygame
import heapq
import time
import sys

from Game_UI import SlidePuzzle

FPS = 60


def main():
    """
    The main function to run the game.
    """
    pygame.init()
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.display.set_caption("8-Puzzle game")
    screen = pygame.display.set_mode((800, 500))
    fpsclock = pygame.time.Clock()
    while True:
        puzzle = SlidePuzzle((3, 3), 160, 5, screen)
        choice = puzzle.selectPlayerMenu("8 Puzzle using A* search")
        if choice == "AI":
            puzzle.shuffle()
            playAIGame(puzzle, fpsclock)
        else:
            puzzle.playHumanGame(fpsclock)


def playAIGame(puzzle, fpsclock):
    """
    Play the game with AI.

    :param puzzle: The puzzle instance
    :param fpsclock: Track time.
    """
    finished = False

    # Save initial conf of game
    conf_init = puzzle.tiles[:]

    # Solve the game with A*
    start = time.time()
    path = iter(solveAI(puzzle))
    print("Exec time", time.time() - start)
    if path is None:
        print("Error, the AI did not find any solution.")
        pygame.quit()
        sys.exit()

    # reset state of puzzle
    puzzle.tiles = conf_init
    for h in range(len(puzzle.tiles)):
        puzzle.tilepos[h] = puzzle.tilePOS[puzzle.tiles[h]]

    while not finished and not puzzle.want_to_quit:
        dt = fpsclock.tick(FPS)
        puzzle.screen.fill((0, 0, 0))
        puzzle.draw()
        puzzle.drawShortcuts(False, None)
        pygame.display.flip()
        puzzle.catchGameEvents(False, lambda: puzzle.switch(next(path), True))
        puzzle.update(dt)

        finished = puzzle.checkGameState(True)


def solveAI(puzzle):
    """
    Implementation of the A* algorithm to solve the 8-puzzle game.

    :param puzzle: The puzzle instance
    :return: The sequence of positions of the blank tile in order to solve the puzzle.
             This corresponds to the path to go from the initial to the winning configuration.
    """
    start = puzzle.tiles
    q = [(0, start, [start[-1]])]
    # We transform q into a priority queue (heapq)
    heapq.heapify(q)
    g_scores = {str(start): 0}

    while len(q) != 0:
        current = heapq.heappop(q)
        puzzle.tiles = current[1]

        if puzzle.isWin():
            print("Found solution:", current[2])
            return current[2]

        for m in moves(puzzle):
            # for all moves, g is the current cost
            g = g_scores[str(current[1])] + 1
            # f is the sum of the current cost + heuristic
            f = g + heuristic(puzzle, m)
            if str(m) not in g_scores or g < g_scores[str(m)]:
                heapq.heappush(q, (f, m, current[2] + [m[-1]]))
                g_scores[str(m)] = g


def moves(puzzle):
    """
    Compute the accessible configurations from the current one with the possible moves.

    :param puzzle: The puzzle instance
    :return: The sets of accessible configurations.
    """
    moves = []
    potential_neighbors = puzzle.adjacent()

    for n in potential_neighbors:
        if puzzle.inGrid(n):
            new_tiles = puzzle.tiles.copy()
            new_tiles[-1] = n
            pos_n = puzzle.tiles.index(n)
            new_tiles[pos_n] = puzzle.getBlank()
            moves.append(new_tiles)
    return moves


def heuristic(puzzle, n):
    """
    Compute the Manhattan distance of all tiles corresponding to the heuristic used in the A* algorithm.

    :param puzzle: The puzzle instance
    :param n: Configuration for which we want to compute the total Manhattan distance.
    :return:  Total Manhattan distances for all tiles in configuration.
    """
    dist = 0
    for i in range(9):
        # Sum of Manhattan distances for all tiles
        dist += abs(n[i][0] - puzzle.winCdt[i][0]) + abs(n[i][1] - puzzle.winCdt[i][1])
    return dist


if __name__ == "__main__":
    main()
