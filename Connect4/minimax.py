from bot import Bot
import copy
from common import MINIMAX, EMPTY, ROW_COUNT, COLUMN_COUNT, WINDOW_LENGTH
import random
import math


class MiniMax(Bot):
    """
    This class is responsible for the Minimax algorithm.
    At each depth, the algorithm will simulate up to 7 boards, each having a piece that has been dropped in a free column. So with depth 1, we will have 7 boards to analyse, with depth 2 : 49 ,...
    Through a system of reward each board will be attributed a score. The Minimax will then either try to minimise or maximise the rewards depending on the depth (odd or even). Indeed, because we are using multiple
    depth, the minimax algorithm will simulate in alternance the possible moves of the current player and the ones of the adversary (creating Min nodes and max nodes). The player that needs to decide where to
    drop a piece on the current board is considered as the maximising player, hence trying to maximise the reward when a max nodes is encountered. The algorithm will also consider that the adversary plays as good as possible (with
    the information available with the depth chosen) and hence try to minimise the reward when possible (minimizing player).
    So after creating all the boards of the tree, at each depth, a board will be selected based on the reward and on the type of nodes (min or max node) starting from the bottom of the tree.
    The final choice is made based on the 7 boards possible with the score updated through the reward procedure describe above.
    Note that the larger the depth, the slower the execution.
    In order to avoid unnecessary exploration of boards, an alpha beta pruning has been implemented.
    """

    def __init__(self, game, depth, pruning=True):
        super().__init__(game, bot_type=MINIMAX, depth=depth, pruning=pruning)

    def drop_piece(self, board, row, col, piece):
        """
        Drop a piece in the board at the specified position
        :param board: board with all the pieces that have been placed
        :param col: one of the row of the board
        :param col: one of the column of the board
        :param piece: 1 or -1 depending on whose turn it is
        """
        board[col][row] = piece

    def get_next_open_row(self, board, col):
        """
        Return the first row which does not have a piece in the specified column (col)
        :param board: board with all the pieces that have been placed
        :param col: one of the column of the board
        :return: row number
        """
        for r in range(ROW_COUNT):
            if board[col][r] == 0:
                return r

    def winning_move(self, board, piece):
        """
        Check if the game has been won
        :param board: board with all the pieces that have been placed
        :param piece: 1 or -1 depending on whose turn it is
        """
        # Check horizontal locations for win
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (
                    board[c][r] == piece
                    and board[c + 1][r] == piece
                    and board[c + 2][r] == piece
                    and board[c + 3][r] == piece
                ):
                    return True

        # Check vertical locations for win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (
                    board[c][r] == piece
                    and board[c][r + 1] == piece
                    and board[c][r + 2] == piece
                    and board[c][r + 3] == piece
                ):
                    return True

        # Check positively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (
                    board[c][r] == piece
                    and board[c + 1][r + 1] == piece
                    and board[c + 2][r + 2] == piece
                    and board[c + 3][r + 3] == piece
                ):
                    return True

        # Check negatively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (
                    board[c][r] == piece
                    and board[c + 1][r - 1] == piece
                    and board[c + 2][r - 2] == piece
                    and board[c + 3][r - 3] == piece
                ):
                    return True
        return False

    def is_terminal_node(self, board):
        """
        Determines wheter the game is finished or not
        :param board: board with all the pieces that have been placed
        :return: boolean that determines wheter the game is finish or not
        """
        return (
            self.winning_move(board, self._game._turn * -1)
            or self.winning_move(board, self._game._turn)
            or self.get_valid_locations(board) is None
        )

    def evaluate_window(self, window, piece):
        """
        Evaluates the score of a portion of the board
        :param window: portion of the board with all the pieces that have been placed
        :param piece: 1 or -1 depending on whose turn it is
        :return: score of the window
        """
        score = 0
        opp_piece = self._game._turn * -1
        if piece == self._game._turn * -1:
            opp_piece = self._game._turn

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 4

        return score

    def score_position(self, board, piece):
        """
        Main function that handles the scoring mechanism.
        Handle the score for the minimax algorithm, the score is computed independently of which piece has just been dropped. This is a global score that looks at the whole board
        :param board: board with all the pieces that have been placed
        :param piece: 1 or -1 depending on whose turn it is
        :return: score of the board
        """

        score = 0
        # Score center column
        center_array = [int(i) for i in list(board[COLUMN_COUNT // 2][:])]
        center_count = center_array.count(piece)
        score += center_count * 3

        # Score Horizontal
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in list(board[:][r])]
            for c in range(COLUMN_COUNT - 3):
                window = row_array[c : c + WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score Vertical
        for c in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(board[c][:])]
            for r in range(ROW_COUNT - 3):
                window = col_array[r : r + WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        # Score posiive sloped diagonal
        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [board[c + i][r + i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for r in range(ROW_COUNT - 3):
            for c in range(COLUMN_COUNT - 3):
                window = [board[c + i][r + 3 - i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score

    def minimax(self, board, depth, alpha, beta, maximizingPlayer, pruning):
        """
        Main function of minimax, called whenever a move is needed.
        Recursive function, depth of the recursion being determined by the parameter depth.
        :param depth: number of iterations the Minimax algorith will run for
            (the larger the depth the longer the algorithm takes)
        :alpha: used for the pruning, correspond to the lowest value of the range values of the node
        :beta: used for the pruning, correspond to the hihest value of the range values of the node
        :maximizingPlayer: boolean to specify if the algorithm should maximize or minimize the reward
        :pruning: boolean to specify if the algorithm uses the pruning
        :return: column where to place the piece
        """
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board)

        if depth == 0:
            return (None, self.score_position(board, self._game._turn))
        elif is_terminal:
            if self.winning_move(board, self._game._turn):
                return (None, math.inf)
            elif self.winning_move(board, self._game._turn * -1):
                return (None, -math.inf)
            else:  # Game is over, no more valid moves
                return (None, 0)

        column = valid_locations[0]

        if maximizingPlayer:
            value = -math.inf
            turn = 1
        else:
            value = math.inf
            turn = -1

        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            b_copy = copy.deepcopy(board)

            self.drop_piece(b_copy, row, col, self._game._turn * turn)
            new_score = self.minimax(
                b_copy, depth - 1, alpha, beta, not maximizingPlayer, pruning
            )[1]

            if maximizingPlayer:
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
            else:
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)

            if pruning:
                if alpha >= beta:
                    break

        return column, value
