""" This file provides a bare-bone Tetris game controller """

# Imports
import random
import TetrisUtils as TUtils
from TetrisSettings import *

import numpy as np

# This is the bare bones of the tetris game
# Does not contain UI, simply a 2D array playing tetris
class Tetris:

    # Each tetris game need to keep track of:
    # - the game
    # - score
    # - fitness
    def __init__(self):
        ##################
        # Game logistics #
        ##################
        # Game over indicator
        self.game_over = True
        # Game board: 2D array of integers representing pieces
        self.board = [] #todo: replace by a numpy array
        self.tile_pool = []
        # Tiles are represented as strings in:
        # ["LINE", "L", "L_REVERSED", "S", "S_REVERSED", "T", "CUBE"]
        self.current_tile = ""
        self.next_tile = ""
        # Current tile location
        self.tile_x = 0
        self.tile_y = 0
        # Current tile shape
        # Save this in order to save tile rotations
        self.tile_shape = []
        self.tetrominoes_number = 0

        ##############
        # Statistics #
        ##############
        self.score = 0.0

        # Make the board playable
        self.reset_game()

    def reset_game(self):
        """ Resets the entire game including statistics """
        self.game_over = False
        self.board = [[0] * GRID_COL_COUNT for _ in range(GRID_ROW_COUNT)]
        self.spawn_tile()
        self.score = 0.0

    def step(self, action: int):
        """
        Advance the game board into the next step

        :param action: index of [NOTHING, L, R, 2L, 2R, ROTATE, SWAP, FAST_FALL, INSTANT_FALL]
        """
        assert action in list(range(8 + 1)), "Invalid action, use 0-8 for actions"
        # If game over, ignore step request until reset() is called
        if self.game_over:
            return
        # Move tile
        if action in [1, 2, 3, 4]:
            self.move_tile((-1 if action in [1, 3] else 1) * (1 if action in [1, 2] else 2))
        # Rotate tile
        elif action == 5:
            self.rotate_tile()
        # Swap current & future tile
        elif action == 6:
            self.swap_tile()
        # Fast fall / instant fall
        elif action in [7, 8]:
            self.drop_tile(instant=(action == 8))

        # Drop tile by 1 grid
        self.drop_tile()

    def spawn_tile(self) -> bool:
        """
        Spawns a new tile from the tile pool
        :return: whether the game is over
        """
        self.current_tile = self.get_next_tile(pop=True)
        self.tile_shape = TILE_SHAPES[self.current_tile][:]
        self.tile_x = int(GRID_COL_COUNT / 2 - len(self.tile_shape[0]) / 2)
        self.tile_y = 0
        self.tetrominoes_number += 1

        # Game over check: game over if new tile collides with existing blocks
        return TUtils.check_collision(self.board, self.tile_shape, (self.tile_x, self.tile_y))

    def generate_tile_pool(self):
        """ Resets the tile pool """
        self.tile_pool = list(TILE_SHAPES.keys())
        random.shuffle(self.tile_pool)
        # np.random.shuffle(self.tile_pool)

    def on_tile_collision(self):
        # Add current tile to board
        for cy, row in enumerate(self.tile_shape):
            for cx, val in enumerate(row):
                if val == 0:
                    continue
                self.board[cy + self.tile_y - 1][min(cx + self.tile_x, 9)] = val

        # Check completed rows
        row_completed = 0
        row_index = 0
        while True:
            if row_index >= len(self.board):
                break
            # If there's any unfilled space
            if 0 in self.board[row_index]:
                row_index += 1
                continue
            # Delete the completed row
            del self.board[row_index]
            # Insert empty row on top
            self.board.insert(0, [0] * GRID_COL_COUNT)
            row_completed += 1

        # Calculate total score
        self.score += MULTI_SCORE_ALGORITHM(row_completed)

        # Spawn next tile
        self.game_over = self.spawn_tile()

    #########################
    # Step Action Functions #
    #########################
    def drop_tile(self, instant=False):
        """
        Drop the current tile by 1 grid; if <INSTANT>, then drop all the way

        :param instant: whether to drop all the way
        """
        if instant:
            # Drop the tile until it collides with existing block(s)
            new_y = TUtils.get_effective_height(self.board, self.tile_shape, (self.tile_x, self.tile_y))
            self.tile_y = new_y + 1
            #self.score += PER_STEP_SCORE_GAIN * (new_y - self.tile_y)
        else:
            # Drop the tile by 1 grid
            self.tile_y += 1
            #self.score += PER_STEP_SCORE_GAIN

        # If doesn't collide, skip next step
        if not instant and not TUtils.check_collision(self.board, self.tile_shape, (self.tile_x, self.tile_y)):
            return
        # Trigger collision event
        self.on_tile_collision()

    def move_tile(self, delta: int):
        """
        Move current tile to the right by <DISTANCE> grids

        :param delta: one of [-2, -1, 1, 2]; negative values will move to the left
        """
        assert delta in [-2, -1, 1, 2], "Invalid move distance"
        new_x = self.tile_x + delta
        new_x = max(0, min(new_x, GRID_COL_COUNT - len(self.tile_shape[0])))  # clamping

        # Cannot "override" blocks AKA cannot clip into existing blocks
        if TUtils.check_collision(self.board, self.tile_shape, (new_x, self.tile_y)):
            return
        # Apply tile properties
        self.tile_x = new_x

    def rotate_tile(self):
        """ Rotate current tile by 90 degrees """
        new_tile_shape = TUtils.get_rotated_tile(self.tile_shape)
        new_x = self.tile_x
        # Out of range detection
        if self.tile_x + len(new_tile_shape[0]) > GRID_COL_COUNT:
            new_x = GRID_COL_COUNT - len(new_tile_shape[0])

        # If collide, disallow rotation
        if TUtils.check_collision(self.board, new_tile_shape, (new_x, self.tile_y)):
            return
        # Apply tile properties
        self.tile_x = new_x
        self.tile_shape = new_tile_shape

    def swap_tile(self):
        """ Swaps current tile with the future one """
        # Get next tile without popping (swapping could fail)
        new_tile = self.get_next_tile(pop=False)
        new_tile_shape = TILE_SHAPES[new_tile][:]  # clone tile shape
        temp_x, temp_y = self.tile_x, self.tile_y

        # Out of range detection
        if temp_x + len(self.tile_shape[0]) > GRID_COL_COUNT:
            temp_x = GRID_COL_COUNT - len(self.tile_shape[0])
        if temp_y + len(self.tile_shape) > GRID_ROW_COUNT:
            temp_y = GRID_ROW_COUNT - len(self.tile_shape)

        # If collide, disallow swapping
        if TUtils.check_collision(self.board, new_tile_shape, (temp_x, temp_y)):
            return
        # Put current tile as the next tile
        self.tile_pool[0] = self.current_tile
        # Apply tile properties
        self.current_tile = new_tile
        self.tile_shape = new_tile_shape
        self.tile_x, self.tile_y = temp_x, temp_y

    #####################
    # Utility Functions #
    #####################
    def get_next_tile(self, pop=False):
        """ Obtains the next tile from the tile pool """
        if not self.tile_pool:
            self.generate_tile_pool()
        self.score += 1
        return self.tile_pool[0] if not pop else self.tile_pool.pop(0)


# Test game board using step action
if __name__ == "__main__":
    tetris = Tetris()
    while True:
        if not tetris.game_over:
            # Print game board
            TUtils.print_board(TUtils.get_board_with_tile(tetris.board, tetris.tile_shape, (tetris.tile_x, tetris.tile_y)))

        # Get user input (q = quit, r = reset)
        # 0-8 = actions
        message = input("Next action (0-8): ")
        if message == "q":
            break
        elif message == "r":
            tetris.reset_game()
            continue

        # Step
        tetris.step(int(message))
