from snake import Snake
import copy as cp
import math
import random as rd
from dna import Dna

from gameModule import GUISnakeGame, TrainingSnakeGame

# from gameThread import *
from os import listdir


class SnakesManager:
    """
    Class whose purpose is to make the snakes plays and create new generations of snakes.
    """

    def __init__(
        self,
        guiSnakeGame,
        nbrSnakes,
        layersSize=None,
        mutationRate=0.01,
        hunger=100,
        survivalProportion=0.1,
    ):
        """
        Constructor.
        guiSnakeGame (GUISnakeGame): Used to create a new game and GUI for a snake
        nbrSnakes (int): The number of snake in each generation
        layersSize (list): A list containing the number of hidden neurons for each layer
        mutationRate (float): The probability for the DNA to mutate
        hunger (int): hunger (int): The starting hunger for each snake
        survivalProportion (float): The proportion of snakes that will replay in the next generation
        """
        self.guiSnakeGame = guiSnakeGame
        self.nbrSnakes = nbrSnakes
        self.layersSize = layersSize
        self.hunger = hunger

        if layersSize is None:
            self.layersSize = [rd.randint(10, 15) for i in range(rd.randint(1, 3))]

        self.snakes = [
            Snake(Dna(layersSize=self.layersSize), hunger) for i in range(nbrSnakes)
        ]
        self.games = [TrainingSnakeGame(None) for i in range(nbrSnakes)]
        self.mutationRate = mutationRate
        self.survivalProportion = survivalProportion
        self.currentSnake = 0
        self.generation = 0
        self.bestGenFitness = 0
        self.bestFitness = 0
        self.bestGenScore = 0
        self.totalGenScore = 0
        self.bestSnake = self.snakes[0]  # The best snake of the generation
        self.bestGenScoreSave = 0

    def train(self):
        # Best game score and fitness so far
        bestScore = -1
        bestFitness = -1
        itEnd = 0

        # Create new generations until the stop condition is satisfied
        while itEnd < 150:
            print(f"Generation {self.generation}, best: {bestScore}")
            self.start_gen()
            self.end_gen()

            currentScore = self.get_best_gen_score()
            currentFitness = self.get_best_gen_fitness()
            self.change_generation()

            # Check if the game score or fitness for this generation improved
            if currentScore <= bestScore and currentFitness <= bestFitness:
                itEnd += 1
            else:
                # Improvement + reset counter
                bestScore = max(bestScore, currentScore)
                bestFitness = max(bestFitness, currentFitness)
                itEnd = 0

    def start_gen(self):
        """
        Creates a new game thread for each snake and make them play.
        """
        for s in self.snakes:
            s.reset_state()

        for snake, game in zip(self.snakes, self.games):
            game.learning_agent = snake
            game.start_run()

            while game.is_alive():
                game.next_tick()

    def end_gen(self):
        """
        Waits for the game threads to end and updates the fitness and game score for this generation.
        """
        for snake, game in zip(self.snakes, self.games):
            fitness = snake.compute_fitness(game.get_score())
            self.bestGenFitness = max([fitness, self.bestGenFitness])
            self.bestFitness = max([fitness, self.bestFitness])
            self.bestGenScore = max([game.get_score(), self.bestGenScore])
            self.totalGenScore += game.get_score()

            if fitness == self.bestGenFitness:
                self.bestSnake = snake

        # Save the weights and biases of the snakes for the new game scores
        files = listdir("./weights/")

        # If this is a new game score
        if str(self.bestGenScore) + ".txt" not in files:
            with open("./weights/" + str(self.bestGenScore) + ".txt", "w") as f:
                toWrite = (
                    str(self.bestSnake.dna.weights)
                    + "\n\n"
                    + str(self.bestSnake.dna.bias)
                )
                f.write(toWrite)

    def show_best_snake(self):
        """
        Show the best snake of the current generation playing the game again.
        The snake replays in a new environment (new spawn position for the snake and the food)
        """
        self.bestSnake.reset_state()
        self.guiSnakeGame.start_run()

        while self.guiSnakeGame.is_alive():
            self.guiSnakeGame.next_tick(self.bestSnake)

    def pick_parents(self, matingSnakes):
        """
        Pick two parents to mate and create a new snake that will participate in the next generation.
        The parents are selected according to their fitness. The higher the fitness is the higher is the
        probability for the snake to be chosen.
        """
        parents = []
        totalFitness = sum([snake.get_fitness() for snake in matingSnakes])

        # Choose two parents depending on their fitness
        for t in range(2):
            r = rd.randint(0, totalFitness - 1)
            i = 0

            for snake in matingSnakes:
                if r < snake.getFitness():
                    parents.append(snake)
                    totalFitness -= snake.getFitness()
                    break

                r -= snake.getFitness()
                i += 1

            matingSnakes.pop(i)

        matingSnakes += parents
        return parents

    def pick_parents_rank(self, matingSnakes):
        """
        Pick two parents to mate and create a new snake that will participate in the next generation.
        The parents are selected according to their rank.
        """
        parents = []
        popSize = len(matingSnakes)
        totalFitness = popSize / 2 * (popSize + 1)

        for t in range(2):
            r = rd.randint(0, totalFitness - 1)
            i = 0
            used = None

            while i < len(matingSnakes):
                if r < popSize - i:
                    parents.append(matingSnakes[i])
                    totalFitness -= popSize
                    popSize -= 1
                    used = i
                    break

                r -= popSize - i
                i += 1

                if i == used:
                    i += 1

        return parents

    def change_generation(self):
        """
        Creates a new generation of snakes.
        """
        newSnakes = sorted(self.snakes, key=lambda x: x.get_fitness(), reverse=True)
        a0 = f"Generation {self.generation}"
        a1 = f"Average fitness for this generation: {sum([snake.get_fitness() for snake in self.snakes])/self.nbrSnakes}"
        a2 = f"Median fitness for this generation: {newSnakes[len(newSnakes)//2].get_fitness()}"
        a3 = f"Best fitness for this generation: {self.bestGenFitness}"
        a4 = f"Best gamescore for this generation: {self.bestGenScore}"
        a5 = f"Average gamescore for this generation: {self.totalGenScore/self.nbrSnakes}"

        # print(a0)
        # print(f"{self.generation}\n{a1}\n{a2}\n{a3}\n{a4}\n{a5}\n\n")

        # Save the data
        with open(
            f"data/{self.nbrSnakes}-{self.layersSize}-{self.mutationRate}-{self.hunger}-{self.survivalProportion}.txt",
            "a+",
        ) as f:
            f.write(f"{self.generation}\n{a1}\n{a2}\n{a3}\n{a4}\n{a5}\n\n")

        # Sort the snakes by their fitness (decreasing)
        newSnakes = newSnakes[: int(self.nbrSnakes * self.survivalProportion)]
        matingSnakes = cp.copy(newSnakes)

        # Generate new snakes
        while len(newSnakes) < self.nbrSnakes:
            # Creates a new snake and add it to the next generation
            parents = self.pick_parents_rank(matingSnakes)
            mutationRate = self.mutationRate
            baby = parents[0].mate(parents[1], mutationRate=mutationRate)
            newSnakes.append(baby)

        # Update
        self.snakes = newSnakes
        self.bestGenFitness = 0
        self.bestGenScore = 0
        self.totalGenScore = 0
        self.generation += 1

    def get_generation(self):
        """
        Returns the generation number (int).
        """
        return self.generation

    def get_best_snake(self):
        """
        Returns the snake that had the highest game score for this generation (Snake).
        """
        return self.bestSnake

    def get_best_gen_score(self):
        """
        Returns the highest game score of this generation (int).
        """
        return self.bestGenScore

    def get_best_gen_fitness(self):
        """
        Returns the highest fitness of this generation (float).
        """
        return self.bestGenFitness