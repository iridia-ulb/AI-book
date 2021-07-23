import math
import os
from copy import copy
from random import random, randint
from Game_UI import SlidePuzzle, WIDTH, HEIGHT
import pygame


def main():
    """
    The main function to run the game.
    """
    pygame.init()
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.display.set_caption("8-Puzzle game")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    fpsclock = pygame.time.Clock()
    while True:
        game = SlidePuzzle((3, 3), 160, 5, screen)
        # Ask if the player wants to play on a trained model or not.
        choice = game.selectPlayerMenu()
        if choice == "AI":
            modelAI = game.selectModel()
            if modelAI != "":
                game.initPlayerAI(AIPlayer(modelAI))
                trainAI = game.playTrainMenu()
                if trainAI:
                    trainingNb = game.selectTrainingNb()
                    if trainingNb != 0:
                        game.trainingAI(trainingNb)
                elif trainAI == False:
                    # game.selectBoard(fpsclock)
                    game.shuffle()
                    game.playAIGame(fpsclock)
        else:
            game.playHumanGame(fpsclock)


class AIPlayer:

    # The integer values of a direction.
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, fileQTable, epsilon=0.5, gamma=0.4, learningSteps=60000):
        """
        Class constructor.

        :param fileQTable:      A string corresponding to the path of the QTable to load.
        :param epsilon:         A double corresponding to the probability of exploring a non-optimal move.
        :param gamma:           A double corresponding to the discount factor of the Bellman equation.
        :param learningSteps:   An integer, during learning, each time the AI plays “learningSteps” games, the
                                learning rate will be divided by 10 (initially the learning rate = 0.1).
        """
        self.nbPlayedGames = 1  # The number of games played by the AI.
        self.epsilon = epsilon
        self.gamma = gamma
        self.learningSteps = learningSteps
        self.currentSolution = []  # The list of moves of the last solution.

        # A permutations of the string "123456789" corresponds to possible state
        # of the game. And in each state, we can make 4 actions [up, right, down, left].
        # The Q-Table is indexed in the lexicographical order of the string representing a state.
        self.qTable = []

        self.qTablePath = fileQTable

        if fileQTable != "" and os.path.isfile(fileQTable):
            self.loadQTable()
        else:
            for i in range(math.factorial(9)):
                self.qTable.append([0.0, 0.0, 0.0, 0.0])

    def getNbGames(self):
        """
        Return the total number of games played by the AI.

        :return:        Integer corresponding to the number of played games.
        """
        return self.nbPlayedGames

    def getNextMove(self):
        """
        Pop the current move of the current solution.

        :return:        Integer corresponding to the direction of the next move.
        """
        return self.currentSolution.pop(0)

    def getQValues(self, state):
        """
        Return the Q-Values of a given state.

        :param state:       A string corresponding to a state of the game.
        :return:            A list of Q-Values corresponding to the 4 possible directoins.
        """
        return self.qTable[findRank(state) - 1].copy()

    def saveQTable(self):
        """
        Saves the QTable in a text file called "QTable.txt".
        """
        f = open(self.qTablePath, "w")
        f.write("{}\n".format(self.nbPlayedGames))
        f.write("{}\n".format(self.epsilon))
        f.write("{}\n".format(self.gamma))
        f.write("{}\n".format(self.learningSteps))
        for i in range(math.factorial(9)):
            for j in range(4):
                f.write("{} ".format(self.qTable[i][j]))
            f.write("\n")
        f.close()

    def loadQTable(self):
        """
        Loads the QTable contained in the "QTable.txt" file.
        """
        f = open(self.qTablePath, "r")
        self.nbPlayedGames = int(f.readline().strip())
        self.epsilon = float(f.readline().strip())
        self.gamma = float(f.readline().strip())
        self.learningSteps = int(f.readline().strip())
        for i in range(math.factorial(9)):
            directions = []
            directionValues = f.readline().split()
            for j in range(4):
                directions.append(float(directionValues[j]))
            self.qTable.append(directions)
        f.close()

    def makeMove(self, currentState, direction):
        """
        Makes the desired move by returning the next state of the game.
        If the move is illegal, the same state is returned.

        :param currentState:    A string corresponding to the current state of the game.
        :param direction:       An integer corresponding to the direction where we want to move the empty tile.
        :return:                A string corresponding to the next state of the game (after the move).
        """
        nextState = None
        emptyTileIndex = currentState.index("9")
        if direction == self.UP:
            if emptyTileIndex >= 3:
                destinationTile = currentState[emptyTileIndex - 3]
                nextState = (
                    currentState[0 : emptyTileIndex - 3]
                    + "9"
                    + currentState[emptyTileIndex - 2 : emptyTileIndex]
                    + destinationTile
                    + currentState[emptyTileIndex + 1 :]
                )
        elif direction == self.DOWN:
            if emptyTileIndex <= 5:
                destinationTile = currentState[emptyTileIndex + 3]
                nextState = (
                    currentState[0:emptyTileIndex]
                    + destinationTile
                    + currentState[emptyTileIndex + 1 : emptyTileIndex + 3]
                    + "9"
                    + currentState[emptyTileIndex + 4 :]
                )
        elif direction == self.LEFT:
            if emptyTileIndex % 3 != 0:
                destinationTile = currentState[emptyTileIndex - 1]
                nextState = (
                    currentState[0 : emptyTileIndex - 1]
                    + "9"
                    + destinationTile
                    + currentState[emptyTileIndex + 1 :]
                )
        else:
            if emptyTileIndex % 3 != 2:
                destinationTile = currentState[emptyTileIndex + 1]
                nextState = (
                    currentState[0:emptyTileIndex]
                    + destinationTile
                    + "9"
                    + currentState[emptyTileIndex + 2 :]
                )
        if nextState == None:
            nextState = currentState
        return nextState

    def selectNewAction(self, currentState):
        """
        Chooses a new action according to the epsilon-greedy selection method.

        :param currentState:    A string corresponding to the current state of the game.
        :return:                An integer corresponding to the new action
                                (i.e. the direction we want to move the empty tile to).
        """
        # Choose a random action with probability epsilon.
        if random() < self.epsilon:
            return randint(0, 3)
        # With probability 1-epsilon, choose the action corresponding to the maximum reward.
        else:
            stateIndex = findRank(currentState) - 1
            maxQValue = max(self.qTable[stateIndex])
            # If multiple actions give the maximum value, we randomly choose one of those maximum actions.
            maxIndexes = []
            for i in range(4):
                if self.qTable[stateIndex][i] == maxQValue:
                    maxIndexes.append(i)
            return maxIndexes[randint(0, len(maxIndexes) - 1)]

    def getAlpha(self):
        """
        Returns the current learning rate as a function of the number of already played games.
        The initial learning rate is 0.1 and it is divided by 10 each time we play "self.learningSteps" games.

        :return:    Double corresponding to the current learning rate.
        """
        return 1 / (
            10 ** ((self.nbPlayedGames + self.learningSteps) / self.learningSteps)
        )

    def playGame(self, state):
        """
        Plays a given game.

        :param state:       A string corresponding to a state of the game.
        """
        oldEpsilon = self.epsilon
        self.epsilon = 0.0
        self.currentSolution = []
        while state != "123456789":
            newAction = self.selectNewAction(state)
            newState = self.playRound(state, newAction)
            if newState != state:
                self.currentSolution.append(newAction)
                state = newState
        self.nbPlayedGames += 1
        self.epsilon = oldEpsilon

    def playRound(self, state, action):
        """
        Moves a tile and updates the Q-Table.

        :param state:       A string corresponding to the current state of the game.
        :param action:      An integer corresponding to the direction where to move the empty tile.
        :return:            A string corresponding to the next state of the game (after the move).
        """
        stateIndex = findRank(state) - 1
        nextState = self.makeMove(state, action)
        if nextState == "123456789":
            reward = 1
        else:
            reward = 0
        self.qTable[stateIndex][action] += self.getAlpha() * (
            reward
            + self.gamma * max(self.qTable[findRank(nextState) - 1])
            - self.qTable[stateIndex][action]
        )
        return nextState

    def generateNStepsGame(self, nbMoves):
        """
        Generates an instance of the game that takes at most "nbMoves" to be solved.

        :param nbMoves:     An integer corresponding to the maximum numbers of moves
                            necessary to solve the game instance.
        :return:            A string corresponding to an instance of the 8Puzzle game.
        """
        initState = "123456789"
        currentState = initState
        # We start by the final state and move away from it.
        while nbMoves > 0:
            newAction = randint(0, 3)
            newState = self.makeMove(currentState, newAction)
            if newState != initState and newState != currentState:
                currentState = newState
                nbMoves -= 1
        return currentState

    def generateGame(self):
        """
        Generates a random instance of the 8Puzzle game.

        :return:    A string corresponding to an instance of the game.
        """
        # We iterate if we created un unsolvable instance of the game.
        solvableGame = False
        while not solvableGame:
            tiles = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            game = []
            for i in range(9):
                newTileIndex = randint(0, 8 - i)
                game.append(tiles.pop(newTileIndex))
            solvableGame = self.isSolvable(game)
        # We convert the state, encoded as a list, into an integer.
        stateAsInt = 0
        for j in range(9):
            stateAsInt += (10 ** (8 - j)) * game[j]
        return str(stateAsInt)

    def initLearning(self, typeGame, nbGames):
        """
        An initial learning of the AI agent. We play a given number of games,
        that can be solved in at least 'typeGame' moves.

        :param typeGame:        Integer indicating that the games can be solved in at least
                                'typegames' moves.
        :param nbGames:         Integer corresponding to the number of games to train the AI on.
        """
        gamesWon = 0
        while gamesWon < nbGames:
            state = self.generateNStepsGame(typeGame)
            nbMoves = 0
            while state != "123456789" and nbMoves < typeGame * 100:
                newAction = self.selectNewAction(state)
                newState = self.playRound(state, newAction)
                if newState != state:
                    state = newState
                    nbMoves += 1
            if state == "123456789":
                gamesWon += 1

    def preTrain(self, nbGames=100):
        """
        Proceeds to the inital training, by playing "nbGames" of each type.

        :param nbGames:     Integer corresponding to the number of games to train the AI on.
        """
        for i in range(31):
            self.initLearning(i + 1, nbGames)

    def isSolvable(self, game):
        """
        Checks if a given game instance is solvable.

        :param game:        A list of integers corresponding to the initial state of the game.
        :return:            A boolean that is True if the instance is solvable.
        """
        inversions = 0
        for i in range(len(game) - 1):
            for j in range(i + 1, len(game)):
                if game[i] > game[j] and game[i] != 9:
                    inversions += 1
        return True if (inversions % 2 == 0 and inversions != 0) else False

    def train(self, nbGames):
        """
        Trains the AI by playing a specified number of games. The games are randomly generated.

        :param nbGames:     An integer corresponding to the number of games to train the AI on.
        :return:            Integer corresponding to total number of moves.
        """
        nbMoves = 0
        for _ in range(nbGames):
            state = self.generateGame()
            while state != "123456789":
                newAction = self.selectNewAction(state)
                state = self.playRound(state, newAction)
                nbMoves += 1
            self.nbPlayedGames += 1
        return nbMoves


def findSmallerInRight(string, start, end):
    """
    Counts the number of caracters that are smaller than
    string[start] and are at the right of it.

    :param string:      Input string.
    :param start:       Integer corresponding to the index of the starting character.
    :param end:         Integer corresponding to the index of the string.
    :return:            Integer corresponding to the number of chars, on the right, that are smaller.
    """
    countRight = 0
    i = start + 1
    while i <= end:
        if string[i] < string[start]:
            countRight = countRight + 1
        i = i + 1
    return countRight


def findRank(string):
    """
    Returns the rank of the given string, considering all the
    possible permutations of the string.

    :param string:      Input string.
    :return:            An integer corresponding to the rank of the input string.
    """
    strLen = len(string)
    mul = math.factorial(strLen)
    rank = 1
    i = 0
    while i < strLen:
        mul = mul / (strLen - i)
        countRight = findSmallerInRight(string, i, strLen - 1)
        rank = rank + countRight * mul
        i = i + 1
    return int(rank)


if __name__ == "__main__":
    main()