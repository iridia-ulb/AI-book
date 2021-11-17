import copy as cp
import random as rd
import pickle
from pathlib import Path
from os import listdir
from dna import Dna
from gameModule import SnakeGame
from genetic_snake import Snake


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
            self.layersSize = [
                rd.randint(10, 15) for i in range(rd.randint(1, 3))
            ]

        self.snakes = [
            Snake(Dna(layersSize=self.layersSize), hunger)
            for i in range(nbrSnakes)
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
            print(
                f"Generation {self.generation}, best: {bestScore}, bestfit: {bestFitness}"
            )
            self.eval_gen()

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

    def eval_gen(self):
        """
        Creates a new game thread for each snake and make them play.
        """
        for snake, game in zip(self.snakes, self.games):
            snake.reset_state()
            game.learning_agent = snake
            game.start_run()

            while game.is_alive():
                game.next_tick()

            fitness = snake.compute_fitness(game.get_score())
            self.bestGenFitness = max([fitness, self.bestGenFitness])
            self.bestFitness = max([fitness, self.bestFitness])
            self.bestGenScore = max([game.get_score(), self.bestGenScore])
            self.totalGenScore += game.get_score()

            if fitness == self.bestGenFitness:
                self.bestSnake = snake

        # Save the weights and biases of the snakes for the new game scores
        files = listdir(Path("weights"))

        # If this is a new game score
        if str(self.bestGenScore) + ".snake" not in files:
            with open(
                Path("weights") / Path(str(self.bestGenScore) + ".snake"), "wb"
            ) as f:
                pickle.dump(
                    (self.bestSnake.dna.weights, self.bestSnake.dna.bias), f
                )

    def show_best_snake(self):
        """
        Show the best snake of the current generation playing the game again.
        The snake replays in a new environment (new spawn position for the snake and the food)
        """
        self.bestSnake.reset_state()
        self.guiSnakeGame.start_run()

        while self.guiSnakeGame.is_alive():
            self.guiSnakeGame.next_tick(self.bestSnake)

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
        # Sort the snakes by their fitness (decreasing)
        newSnakes = sorted(
            self.snakes, key=lambda x: x.get_fitness(), reverse=True
        )
        a1 = f"Average fitness for this generation: {sum([snake.get_fitness() for snake in self.snakes])/self.nbrSnakes}"
        a2 = f"Median fitness for this generation: {newSnakes[len(newSnakes)//2].get_fitness()}"
        a3 = f"Best fitness for this generation: {self.bestGenFitness}"
        a4 = f"Best gamescore for this generation: {self.bestGenScore}"
        a5 = f"Average gamescore for this generation: {self.totalGenScore/self.nbrSnakes}"

        print(f"{self.generation}\n{a1}\n{a2}\n{a3}\n{a4}\n{a5}\n\n")

        # Select best snakes
        newSnakes = newSnakes[: int(self.nbrSnakes * self.survivalProportion)]

        # Generate new snakes
        while len(newSnakes) < self.nbrSnakes:
            # Creates a new snake and add it to the next generation
            parents = self.pick_parents_rank(newSnakes)
            baby = parents[0].mate(parents[1], mutationRate=self.mutationRate)
            newSnakes.append(baby)

        # Update
        self.snakes = newSnakes
        self.bestGenFitness = 0
        self.bestGenScore = 0
        self.totalGenScore = 0
        self.generation += 1

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


class TrainingSnakeGame(SnakeGame):
    def __init__(self, learning_agent):
        super(TrainingSnakeGame, self).__init__()
        self.learning_agent = learning_agent
        self.score = 0

    def next_tick(self):
        if self.is_alive():
            # print("Snake is alive, state: ", self.get_state())
            self.set_next_move(
                self.learning_agent.choose_next_move(self.get_state())
            )
            # print(self.next_move)
            if self.foodEaten:
                self.learning_agent.eat()
            return self.move_snake()

        return self.get_state()

    def get_score(self):
        return self.score
