import random
import math
from common import (
    ROW_COUNT,
    COLUMN_COUNT,
    MINIMAX,
    MONTE_CARLO,
    RANDOM,
    RANDOM_IMPR,
    Observer,
)

YELLOW_PLAYER = 1
RED_PLAYER = -1

PLAYERS = {1: "Yellow", -1: "Red"}


class Bot(Observer):
    """
    This class handles the different bots that were used.
    It includes a Random Bot, an Improved Random Bot, the MCTS bot,
    and the MiniMax bot.
    """

    def __init__(
        self, game, bot_type=None, depth=None, iteration=None, pruning=True
    ):
        """
        Constructor of the Bot class.

        :param game: corresponding Connect4Game instance
        :param bot_type: specifies the bot (MCTS, MiniMax, Random, ...)
        :param depth: depth used in the Minimax algorithm if the Minimax bot is used
        :param iteration: number of iterations used in the MCTS algorithm in case the MCTS bot is used
        :param pruning: boolean used for the pruning in the Minimax algorithm if the Minimax bot is used
        """
        self._game = game
        # Bot type determines how the bot picks his moves
        self._type = bot_type
        if self._type == MINIMAX:
            self._depth = depth
            self._pruning = pruning
        elif self._type == MONTE_CARLO:
            self._iteration = iteration

    def __repr__(self):
        return self._type

    def update(self, obj, event, *argv):
        print(obj)

    def make_move(self):
        """
        Picks the column in which the bot should place the next disc.
        The considered moving options depend on the bot type.

        :return: the column number where the bot should play the next move
        """
        # print(PLAYERS[self._game._turn] + " is about to play :")
        column = None
        # In case the bot type is RANDOM, the bot checks for winning moves, and if there aren't,
        # then picks a valid random move.
        if self._type == RANDOM:
            win_col = self.get_winning_move()
            if win_col is not None:
                column = win_col
            else:
                column = self.get_random_move()
        # In case the bot type is RANDOM IMPROVED, the bot checks for winning moves, and if there aren't,
        # then checks if there is any move that blocks a direct winning move for the opponent.
        # If there is no such move, it picks a valid random move.
        elif self._type == RANDOM_IMPR:
            win_col = self.get_winning_move()
            if win_col is not None:
                # print("Winning column :", win_col)
                column = win_col
            else:
                def_move = self.get_defensive_move()
                if def_move is not None:
                    # print("Defensive column :", def_move)
                    column = def_move
                else:
                    column = self.get_random_move()
                    # print("Random move", column)
        elif self._type == MINIMAX:
            column, minimax_score = self.minimax(
                self._game._board,
                self._depth,
                -math.inf,
                math.inf,
                True,
                self._pruning,
            )
            # print(column)
        elif self._type == MONTE_CARLO:
            o = Node(self._game.copy_state())
            column = self.monte_carlo_tree_search(self._iteration, o, 2.0)
        else:
            column = 0

        # print("-------------------------")
        self._game.place(column)

    def get_winning_move(self):
        """
        Checks whether there is a winning column available for the next
        move of the bot.

        :return: winning column
        """
        column = None
        for c_win in range(self._game._cols):
            for r in range(self._game._rows):
                if self._game._board[c_win][r] == 0:
                    self._game._board[c_win][r] = self._game._turn
                    is_winner = self._game.check_win((c_win, r))
                    self._game._board[c_win][r] = 0
                    if is_winner:
                        column = c_win
                        return column
                    break
        return column

    def get_valid_locations(self, board):
        """
        Returns all the valid columns where the player can play, aka the columns
        that are not full

        :param board: actual state of the game, board of the game
        :return: list of all valid column indices
        """
        free_cols = []
        for i in range(COLUMN_COUNT):
            if board[i][ROW_COUNT - 1] == 0:
                free_cols.append(i)
                # print()
        if len(free_cols) == 0:
            return None
        return free_cols

    def get_random_move(self):
        """
        Picks a valid random column where the bot can play his next move.

        :return: valid random column
        """
        free_cols = self.get_valid_locations(self._game._board)
        column = random.choice(free_cols)
        return column

    def get_defensive_move(self):
        """
        Checks whether the bot could play a move that blocks a direct winning
        move from the opponent.

        :return: column to be played to avoid losing immediatly
        """
        column = None
        for c_win in range(self._game._cols):
            for r in range(self._game._rows):
                if self._game._board[c_win][r] == 0:
                    self._game._board[c_win][r] = -1 * self._game._turn
                    is_winner = self._game.check_win((c_win, r))
                    self._game._board[c_win][r] = 0
                    if is_winner:
                        column = c_win
                        return column
                    break
        return column


class Node:
    """
    This class is used to represent nodes of the tree of boards used during
    Monte-Carlo Tree Search.
    """

    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state  # Instance of Connect4Game
        self.children = []
        self.children_moves = []
        self.parent = parent

    def add_child(self, child_state, move):
        """
        Add a child to the current node.

        :param child_state: state of the child to add
        :param move: move to do to get to the newly added child
        """
        child = Node(child_state, parent=self)
        self.children.append(child)
        self.children_moves.append(move)

    def update(self, reward):
        """
        Update the node's reward (indicates how good a certain node is
        according to the MCTS algorithm)

        :param reward: reward to be added to the node
        """
        self.reward += reward
        self.visits += 1

    def fully_explored(self):
        """
        Checks if the node is fully explored (which means we can not add
        any more children to this node)

        :return: True of False depending on if it is fully epxlored or not
        """
        if len(self.children) == len(self.state.get_valid_locations()):
            return True
        return False
