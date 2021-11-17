import copy
import sys
import time
import numpy as np
import random
import pygame
from sudoku_alg import valid


class Population:
    """
    A Population is an ensemble of chromosomes.
    """

    def __init__(self, initial_board):
        self.chromosomes = []
        self.initial_board = initial_board

    def generate_initial_population(self, population_size):
        """
        Generate an ensemble of different chromosomes to start with.
        Each chromosome corresponds to a possible candidate solution to the sudoku problem.
        Each chromosome starts with rows that do not contain any duplicate and this is kept for all the generations.
        """
        non_existing_numbers_matrix_base = []

        for row in range(9):
            # Keep non zero values
            existing_numbers = self.initial_board[row][
                self.initial_board[row] != 0
            ]
            non_existing_numbers = [
                x for x in range(1, 10) if x not in existing_numbers
            ]
            non_existing_numbers_matrix_base.append(non_existing_numbers)

        for candidate in range(population_size):
            non_existing_numbers_matrix = copy.deepcopy(
                non_existing_numbers_matrix_base
            )
            initial_board_copy = self.initial_board.copy()
            for row in range(9):
                for column in range(9):

                    if initial_board_copy[row][column] != 0:
                        continue

                    random_index = random.randint(
                        0, len(non_existing_numbers_matrix[row]) - 1
                    )
                    initial_board_copy[row][
                        column
                    ] = non_existing_numbers_matrix[row][random_index]
                    del non_existing_numbers_matrix[row][random_index]

            self.chromosomes.append(
                Chromosome(initial_board_copy, self.initial_board)
            )
        self.evaluate_fitness_scores()

    def crossover(self, chromosome1, chromosome2):
        """
        Exchange a rows between chromosomes. The rows that can be exchanged must correspond to the same index.
        e.g row 1 from the first chromosome can only be exchange with row 1 of the second chromosome.
        """
        index_rows_to_swap = random.choices(range(9), k=random.randint(1, 4))
        for row in index_rows_to_swap:
            chromosome1.board[row], chromosome2.board[row] = (
                chromosome2.board[row],
                chromosome1.board[row],
            )

    def evaluate_fitness_scores(self):
        """
        Evaluate the fitness score of each chromosome in the population.
        """
        for chromosome in self.chromosomes:
            chromosome.evaluate_fitness_score()

    def sort_chromosomes_on_fitness_score(self):
        """
        Sort the chromosomes in decreasing order of their fitness score.
        """
        self.chromosomes.sort(key=lambda x: x.score, reverse=True)


class Chromosome:
    """
    A chromosome is a candidate solution to the sudoku problem.
    """

    def __init__(self, board, initial_board):
        self.board = board
        self.initial_board = initial_board
        self.score = -1000

    def get_nb_of_duplicates_column(self, column_number):
        """
        Return the quantity of duplicate numbers in a particular column of the candidate
        """
        return 9 - len(
            set([self.board[row][column_number] for row in range(9)])
        )

    def get_nb_of_duplicates_block(self, block_number):
        """
        Return the quantity of duplicates numbers in a particular block of the candidate
        """
        block = set()
        start_row = block_number // 3 * 3
        start_column = block_number % 3 * 3

        for row in range(3):
            for column in range(3):
                block.add(self.board[start_row + row][start_column + column])

        return 9 - len(block)

    def evaluate_fitness_score(self):
        """
        The fitness score is computed as the opposite of the sum of duplicate numbers of each columns and each block of the candidate.
        The quantity of duplicates in each row is not used since this amount always stays at 0
        """
        self.score = 0
        for i in range(9):
            self.score -= self.get_nb_of_duplicates_block(i)
            self.score -= self.get_nb_of_duplicates_column(i)

    def apply_mutation(self):
        """
        A mutation consists in the swap of different random rows of the candidate of different non-blocken element
        in a row. This approach keeps the consistency of the rows.
        """

        index_to_mut = random.choices(range(9), k=random.randint(1, 5))
        for row in index_to_mut:
            # If there are less than two ungiven numbers, pass.
            if (self.initial_board[row] == 0).sum() < 2:
                # print("mut", self.board[row], self.board[row]==0)
                continue

            for _ in range(random.randint(1, 2)):
                first_index = random.randint(0, 8)
                while self.initial_board[row][first_index] != 0:
                    first_index = random.randint(0, 8)

                second_index = random.randint(0, 8)
                while (
                    self.initial_board[row][second_index] != 0
                    or second_index == first_index
                ):
                    second_index = random.randint(0, 8)

                self.board[row][first_index], self.board[row][second_index] = (
                    self.board[row][second_index],
                    self.board[row][first_index],
                )


class Genetic_Algorithm:
    """
    The genetic algorithm will create a random population for the first generation.
    For all generations, crossover between random chromosomes is done.
    With a low probability, a mutation is applied to a chromosome.
    For all generations, Elitism selection is applied, the best solutions are kept intact for the next generation.
    For all generations, selection is applied, only the best solutions are kept and some few random solutions.
    The genetic algorithm stops when for a generation a perfect chromosome is found, the perfect chromosome owns a score of 0.
    Hence, the genetic algorithm tries to maximize the score of the chromosomes of each generation until finding a score of 0.
    """

    def __init__(self, max_nb_gen_with_same_score, board):
        self.population = Population(board)
        self.board = board
        self.generation_size = 10000
        self.elitism_percentage = 0.08
        self.best_score = -1000
        self.nb_gen_with_same_score = 0
        self.max_nb_gen_with_same_score = (81 - max_nb_gen_with_same_score) // 2
        self.population.generate_initial_population(self.generation_size)
        self.population.sort_chromosomes_on_fitness_score()

    def selection(self, elite_chromosomes, new_chromosomes_from_crossover):
        """
        Select which chromosomes are kept in the generation to be passed to the next generation.
        Elitism makes the selection keep the best chromosomes before the modification of the current generation.
        We keep the best chromosomes given by the crossover and some from the mutation operation.
        We keep a small amount of random chromosomes.
        """
        new_chromosomes_from_crossover.sort(key=lambda x: x.score, reverse=True)

        next_population_chromosomes = (
            elite_chromosomes
            + new_chromosomes_from_crossover[
                0 : int(
                    (1 - self.elitism_percentage - 0.1) * self.generation_size
                )
            ]
        )
        new_chromosomes_from_crossover_other_random = (
            new_chromosomes_from_crossover[
                int(
                    (1 - self.elitism_percentage - 0.1) * self.generation_size
                ) :
            ]
        )
        random.shuffle(new_chromosomes_from_crossover_other_random)

        next_population_chromosomes = (
            next_population_chromosomes
            + new_chromosomes_from_crossover_other_random[
                0 : int(0.05 * self.generation_size)
            ]
        )

        random_population = Population(self.board)
        random_population.generate_initial_population(
            int(0.05 * self.generation_size)
        )

        self.population.chromosomes = (
            next_population_chromosomes + random_population.chromosomes
        )
        self.population.sort_chromosomes_on_fitness_score()

    def compute_score_and_escape_from_local_optima(self):
        """
        If the score stays the same in a multiple number of generation, the genetic algorithm is blocked in a local optima.
        It will hence restart from the beginning with a new population of random solutions.
        """
        if self.population.chromosomes[0].score == self.best_score:
            self.nb_gen_with_same_score += 1
            if self.nb_gen_with_same_score == self.max_nb_gen_with_same_score:
                self.population.chromosomes = []
                self.population.generate_initial_population(
                    self.generation_size
                )
                self.population.sort_chromosomes_on_fitness_score()
        else:
            self.nb_gen_with_same_score = 0
        self.best_score = self.population.chromosomes[0].score

    def generate_next_generation(self):
        """
        Generate the future generation by applying Elitism, mutation and selection.
        """
        print("best score: ", self.population.chromosomes[0].score)

        # Elitism: Keep the best results from previous generation
        elite_chromosomes = self.population.chromosomes[
            0 : int(self.generation_size * self.elitism_percentage)
        ]
        random.shuffle(elite_chromosomes)
        new_chromosomes_from_crossover = []

        # Generate double the population size but we only need population_size - population_size * elitism_percentage
        for _ in range(self.generation_size):  # was *2
            (
                new_chromosome1,
                new_chromosome2,
            ) = copy.deepcopy(random.sample(self.population.chromosomes, k=2))

            self.population.crossover(new_chromosome1, new_chromosome2)

            # 25% chance to apply mutation on a chromosome from the crossover
            if random.randint(0, 9) < 5:
                new_chromosome1.apply_mutation()

            new_chromosome1.evaluate_fitness_score()
            new_chromosome2.evaluate_fitness_score()

            new_chromosomes_from_crossover.append(new_chromosome1)
            new_chromosomes_from_crossover.append(new_chromosome2)

        self.selection(elite_chromosomes, new_chromosomes_from_crossover)

        self.compute_score_and_escape_from_local_optima()
        return self.best_score == 0


class GeneticSolver:
    def __init__(self, board, startTime):
        self.board = board
        self.max_nb_gen_with_same_score = 0
        for row in range(9):
            for column in range(9):
                if self.board.board[row][column] != 0:
                    self.max_nb_gen_with_same_score += 1
        self.startTime = startTime

    def visualSolve(self, wrong):
        """Solve the board using a genetic algorithm"""
        genetic_alg = Genetic_Algorithm(
            self.max_nb_gen_with_same_score, self.board.board
        )
        generation = 0
        finished = False
        while not finished:
            for event in pygame.event.get():
                # so that touching anything doesn't freeze the screen
                if event.type == pygame.QUIT:
                    sys.exit()
            finished = genetic_alg.generate_next_generation()
            generation += 1
            pygame.display.set_caption("Generation {}".format(generation))
            best_board = genetic_alg.population.chromosomes[0].board
            for i in range(9):
                for j in range(9):
                    self.board.tiles[i][j].value = best_board[i][j]
                    self.board.tiles[i][j].incorrect = not valid(
                        best_board, (i, j), best_board[i][j]
                    )

                self.board.redraw(
                    {}, wrong, time.time() - self.startTime, generation
                )

            self.board.redraw(
                {}, wrong, time.time() - self.startTime, generation
            )
