# Imports
from typing import *
from Tetris import Tetris
import TetrisUtils as TUtils
from TetrisSettings import *

import random
import numpy as np

"""
In this file is defined the Agent classes. 
The Base Agent is just an abstract class to implement
The Genetic Agent is our implementation of a genetic agent
The only thing that we kept from the original class source code is the calculate action function
The Trained Agent is a class used to create already trained agents
"""
MUTATION_RATE = 0.1


class BaseAgent:
    """ The framework of an agent class, all agents should inherit this class """

    def __init__(self):
        self.action_queue = []

    def get_action(self, tetris: Tetris) -> int:
        if len(self.action_queue) == 0:
            self.action_queue = self.calculate_actions(tetris.board, tetris.tile_shape,
                                                       TILE_SHAPES[tetris.get_next_tile()],
                                                       (tetris.tile_x, tetris.tile_y))
        return self.action_queue.pop(0)

    def calculate_actions(self, board, current_tile, next_tile, offsets) -> List[int]:
        """
        Get best actions from the current board and tile situation

        :param board: Tetris board matrix (2D list)
        :param current_tile: current tile shape (2D list)
        :param next_tile: next tile shape (2D list)
        :param offsets: x, y offsets of the current tile (int, int)
        :return: list of actions to take, actions will be executed in order
        """
        # Overridden by sub-classes
        return [0]


class RandomAgent(BaseAgent):
    """ Agent that randomly picks actions """

    def calculate_actions(self, board, current_tile, next_tile, offsets):
        return [np.random.randint(0, 8) for _ in range(10)]  # np


class GeneticAgent(BaseAgent):
    """ Agent that uses genetics to predict the best action """

    def __init__(self, weigth_to_consider=[0, 1, 2, 3]):
        super().__init__()

        self.weight_array = []

        self.weight_holes = TUtils.random_weight()
        self.weight_array.append(self.weight_holes)

        self.weight_height = TUtils.random_weight()
        self.weight_array.append(self.weight_height)

        self.weight_bumpiness = TUtils.random_weight()
        self.weight_array.append(self.weight_bumpiness)

        self.weight_line_clear = TUtils.random_weight()
        self.weight_array.append(self.weight_line_clear)

        # additional heuristics
        self.weight_hollow_columns = TUtils.random_weight()
        self.weight_array.append(self.weight_hollow_columns)

        self.weight_row_transition = TUtils.random_weight()
        self.weight_array.append(self.weight_row_transition)

        self.weight_col_transition = TUtils.random_weight()
        self.weight_array.append(self.weight_col_transition)

        self.weight_pit_count = TUtils.random_weight()
        self.weight_array.append(self.weight_pit_count)

        self.weight_to_consider = weigth_to_consider

    def get_fitness(self, board):
        """ Utility method to calculate fitness score """

        future_board, rows_cleared = TUtils.get_board_and_lines_cleared(board)

        heuristics = [0 for i in range(8)]

        if 0 in self.weight_to_consider:
            heuristics[0] = TUtils.get_hole_count(future_board)

        if 1 in self.weight_to_consider:
            heuristics[1] = TUtils.get_aggregate_height(future_board)

        if 2 in self.weight_to_consider:
            heuristics[2] = TUtils.get_bumpiness(future_board)

        if 3 in self.weight_to_consider:
            heuristics[3] = rows_cleared

        if 4 in self.weight_to_consider:
            heuristics[4] = TUtils.get_hollow_column_count(future_board)

        if 5 in self.weight_to_consider:
            heuristics[5] = TUtils.get_row_transition(future_board)

        if 6 in self.weight_to_consider:
            heuristics[6] = TUtils.get_col_transition(future_board)

        if 7 in self.weight_to_consider:
            heuristics[7] = TUtils.get_pit_count(future_board)

        score = 0
        for index in self.weight_to_consider:
            score += self.weight_array[index] * heuristics[index]

        return score

    def breed(self, agent):
        """
        "Breed" with another agent to produce a "child"

        :param agent: the other parent agent
        :return: "child" agent
        """

        child = GeneticAgent(self.weight_to_consider)

        self.crossover(agent, child)
        self.mutate_genes(child)

        return child

    def crossover(self, agent, child):
        """
        Crossover the genes of the current agent with another one and modify the genes of the child accordingly
        :param agent: the other agent with which the current is breed
        :param child: the child
        """
        for index in self.weight_to_consider:
            if random.getrandbits(1):
                child.weight_array[index] = self.weight_array[index]
            else:
                child.weight_array[index] = agent.weight_array[index]

    def mutate_genes(self, child):
        """
        Apply the mutation of the genes of the child
        :param child: the child on which to apply the mutation
        """
        for index in self.weight_to_consider:
            if np.random.random() < MUTATION_RATE:
                child.weight_array[index] = TUtils.random_weight()

    # Overrides parent's "abstract" method
    def calculate_actions(self, board, current_tile, next_tile, offsets) -> List[int]:
        """
        Calculate action sequence based on the agent's prediction

        :param board: the current Tetris board
        :param current_tile: the current Tetris tile
        :param next_tile: the next Tetris tile (swappable)
        :param offsets: the current Tetris tile's coordinates
        :return: list of actions (integers) that should be executed in order
        """
        best_fitness = -9999
        best_tile_index = -1
        best_rotation = -1
        best_x = -1

        tiles = [current_tile, next_tile]
        # 2 tiles: current and next (swappable)
        for tile_index in range(len(tiles)):
            tile = tiles[tile_index]
            # Rotation: 0-3 times (4x is the same as 0x)
            for rotation_count in range(0, 4):
                # X movement
                for x in range(0, GRID_COL_COUNT - len(tile[0]) + 1):
                    new_board = TUtils.get_future_board_with_tile(board, tile, (x, offsets[1]), True)
                    fitness = self.get_fitness(new_board)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_tile_index = tile_index
                        best_rotation = rotation_count
                        best_x = x
                # Rotate tile (prep for next iteration)
                tile = TUtils.get_rotated_tile(tile)

        ##################################################################################
        # Obtained best stats, now convert them into sequences of actions
        # Action = index of { NOTHING, L, R, 2L, 2R, ROTATE, SWAP, FAST_FALL, INSTA_FALL }
        actions = []
        if tiles[best_tile_index] != current_tile:
            actions.append(ACTIONS.index("SWAP"))
        for _ in range(best_rotation):
            actions.append(ACTIONS.index("ROTATE"))
        temp_x = offsets[0]
        while temp_x != best_x:
            direction = 1 if temp_x < best_x else -1
            magnitude = 1 if abs(temp_x - best_x) == 1 else 2
            temp_x += direction * magnitude
            actions.append(ACTIONS.index(("" if magnitude == 1 else "2") + ("R" if direction == 1 else "L")))
        actions.append(ACTIONS.index("INSTA_FALL"))
        return actions


class TrainedAgent(GeneticAgent):
    """
    Define an already trained agent
    """

    def __init__(self, precomputed_weigths, weigth_to_consider):
        super().__init__()
        self.weight_array = [weight for weight in precomputed_weigths]
        self.weight_to_consider = weigth_to_consider
