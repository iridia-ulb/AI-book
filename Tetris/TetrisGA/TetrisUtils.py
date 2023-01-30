from copy import deepcopy
from typing import List
import numpy as np

from TetrisSettings import *


###########################
# Board Helper Algorithms #
###########################
def check_collision(board, tile_shape, offsets):
    for cy, row in enumerate(tile_shape):
        for cx, val in enumerate(row):
            if val == 0:
                continue
            try:
                if board[cy + offsets[1]][cx + offsets[0]]:
                    return True
            except IndexError:
                return True
    return False


def get_effective_height(board, tile, offsets):
    offset_x, offset_y = offsets
    while not check_collision(board, tile, (offset_x, offset_y)):
        offset_y += 1
    return offset_y - 1


def get_board_with_tile(board, tile, offsets, flattened=False):
    # Make a copy
    board = deepcopy(board)
    # If flatten, change all numbers to 0/1
    if flattened:
        board = [[int(bool(val)) for val in row] for row in board]
    # Add current tile (do not flatten)
    for y, row in enumerate(tile):
        for x, val in enumerate(row):
            if val != 0:
                board[y + offsets[1]][x + offsets[0]] = val
    return board


def get_future_board_with_tile(board, tile, offsets, flattened=False):
    return get_board_with_tile(board, tile, (offsets[0], get_effective_height(board, tile, offsets)), flattened)


################
# Misc Helpers #
################
def print_board(board):
    print("Printing debug board")
    for i, row in enumerate(board):
        print("{:02d}".format(i), row)


def get_rotated_tile(tile):
    return list(zip(*reversed(tile)))


def get_color_tuple(color_hex):
    if color_hex is None:
        color_hex = "11c5bf"
    color_hex = color_hex.replace("#", "")
    return tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))


######################
# Fitness Algorithms #
######################
# Reference to https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

# Get height of each column
def get_col_heights(board):
    heights = [0] * GRID_COL_COUNT
    cols = list(range(GRID_COL_COUNT))
    for neg_height, row in enumerate(board):
        for i, val in enumerate(row):
            if val == 0 or i not in cols:
                continue
            heights[i] = GRID_ROW_COUNT - neg_height
            cols.remove(i)
    return heights


# Count of empty spaces below covers
def get_hole_count(board):
    holes = 0
    cols = [0] * GRID_COL_COUNT
    for neg_height, row in enumerate(board):
        height = GRID_ROW_COUNT - neg_height
        for i, val in enumerate(row):
            if val == 0 and cols[i] > height:
                holes += 1
                continue
            if val != 0 and cols[i] == 0:
                cols[i] = height
    return holes


# Get the unevenness of the board
def get_bumpiness(board):
    bumpiness = 0
    heights = get_col_heights(board)
    for i in range(1, GRID_COL_COUNT):
        bumpiness += abs(heights[i - 1] - heights[i])
    return bumpiness


# Get the aggregate height of all the columns
def get_aggregate_height(board: List) -> int:
    aggregate_height = 0
    column_checked = []
    for row in range(GRID_ROW_COUNT):
        for col in range(GRID_COL_COUNT):
            if not (col in column_checked) and board[row][col] != 0:
                height = len(board) - row
                aggregate_height += height
                column_checked.append(col)

    return aggregate_height


# Get the number of columns containing at least one hole
def get_hollow_column_count(board):
    nb_hollow_columns = 0
    for col in range(GRID_COL_COUNT):
        is_started = False
        for row in range(GRID_ROW_COUNT):
            if not is_started:
                if board[row][col] != 0:
                    is_started = True
            else:
                if board[row][col] == 0:
                    nb_hollow_columns += 1
                    break
    return nb_hollow_columns

# Get the number of row transition, meaning the number of times we got from occupied cell to unoccupied cell
# when reading row by row.
def get_row_transition(board):
    nb_row_transition = 0
    for row in range(GRID_ROW_COUNT-1, -1, -1):
        is_empty = True if board[row][0] == 0 else False
        is_line_empty = True
        for col in range(GRID_COL_COUNT):
            if board[row][col] == 0 and not is_empty:
                nb_row_transition += 1
                is_empty = True
            elif board[row][col] != 0 and is_empty:
                nb_row_transition += 1
                is_empty = False
                is_line_empty = False
        if is_line_empty:
            break
    return nb_row_transition


# Get the number of column transition, meaning the number of times we got from occupied cell to unoccupied cell
# when reading column by column.
def get_col_transition(board):
    nb_col_transition = 0
    for col in range(GRID_COL_COUNT):
        is_empty = True if board[0][col] == 0 else False
        for row in range(GRID_ROW_COUNT):
            if board[row][col] == 0 and not is_empty:
                nb_col_transition += 1
                is_empty = True
            elif board[row][col] != 0 and is_empty:
                nb_col_transition += 1
                is_empty = False
    return nb_col_transition

# Get the number of empty column
def get_pit_count(board):
    nb_pit = GRID_COL_COUNT
    for col in range(GRID_COL_COUNT):
        for row in range(GRID_ROW_COUNT):
            if board[row][col] != 0:
                nb_pit -= 1
                break
    return nb_pit

# Get potential lines cleared
# WARNING: MODIFIES BOARD!!!
def get_board_and_lines_cleared(board):
    score_count = 0
    row = 0
    while True:
        if row >= len(board):
            break
        if 0 in board[row]:
            row += 1
            continue
        # Delete the "filled" row
        del board[row]
        # Insert empty row at top
        board.insert(0, [0] * GRID_COL_COUNT)
        score_count += 1
    return board, score_count

# Compute a random number between -1 and 1 to be used as weight in an agent's configuration
def random_weight():
    return np.random.uniform(-1, 1)