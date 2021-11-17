from copy import deepcopy
import random
import pygame
import pygame.gfxdraw
from bot import Bot
from common import (
    Event,
    MONTE_CARLO,
    MINIMAX,
    ROW_COUNT,
    COLUMN_COUNT,
    SQUARE_SIZE,
    Observable,
    Observer,
)
from monte_carlo import MonteCarlo
from minimax import MiniMax

# Graphical size settings
DISC_SIZE_RATIO = 0.8

# Colours
BLUE_COLOR = (23, 93, 222)
YELLOW_COLOR = (255, 240, 0)
RED_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (19, 72, 162)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)


class Connect4Game(Observable):
    def __init__(
        self,
        player1,
        player2,
        rows=6,
        cols=7,
        iteration=None,
        depth1=None,
        depth2=None,
        pruning1=True,
        pruning2=True,
    ):
        """
        Constructor of the Connect4Game class.

        :param player1: first player, can be a bot of any type
        :param player2: second player, can be a bot of any type
        :param rows: number of rows in the game
        :param cols: number of columns in the game
        :param iteration: number of iterations used by the players using MCTS
        :param depth1: depth used in the MiniMax algorithm of player1, it is uses MiniMax
        :param depth2: depth used in the MiniMax algorithm of player2, it is uses MiniMax
        """
        super().__init__()
        self._rows = rows
        self._cols = cols
        self._board = None
        self._turn = None
        self._won = None
        self._round = 0
        self.bot = None
        self.moves = {1: [], -1: []}
        self.reset_game()
        if player1 == MONTE_CARLO:
            self._player1 = MonteCarlo(self, iteration=iteration)
        elif player1 == MINIMAX:
            self._player1 = MiniMax(self, depth=depth1, pruning=pruning1)
        else:
            self._player1 = Bot(self, bot_type=player1)
        if player2 == MONTE_CARLO:
            self._player2 = MonteCarlo(self, iteration=iteration)
        elif player2 == MINIMAX:
            self._player2 = MiniMax(self, depth=depth2, pruning=pruning2)
        else:
            self._player2 = Bot(self, bot_type=player2)
        self.last_move = None

    def reset_game(self):
        """
        Resets the game state (board and variables)
        """
        # print("reset")
        self._board = [
            [0 for _ in range(self._rows)] for _ in range(self._cols)
        ]
        self._starter = random.choice([-1, 1])
        self._turn = self._starter
        # (self._turn)
        self._won = None
        self.notify(Event.GAME_RESET)

    def place(self, c):
        """
        Tries to place the playing colour on the c-th column

        :param c: column to place on
        :return: position of placed colour or None if not placeable
        """
        # print(self._board)
        for r in range(self._rows):
            if self._board[c][r] == 0:
                self._board[c][r] = self._turn
                self.last_move = [c, r]
                self.notify(Event.PIECE_PLACED, (c, r))
                self.moves[self._turn].append(c)

                exists_winner = self.check_win((c, r))
                if exists_winner:
                    b = 0
                    if (
                        self._turn == self._starter
                    ):  # Winner is the player that started
                        b = 1
                    self._won = self._turn
                    self.notify(Event.GAME_WON, self._won)

                if self._turn == 1:
                    self._turn = -1
                else:
                    self._turn = 1

                self._round = self._round + 1
                return c, r
        return None

    def check_win(self, pos):
        """
        Checks for win/draw from newly added disc

        :param pos: position from which to check the win
        :return: player number if a win occurs, 0 if a draw occurs, None otherwise
        """
        c = pos[0]
        r = pos[1]
        player = self._board[c][r]

        min_col = max(c - 3, 0)
        max_col = min(c + 3, self._cols - 1)
        min_row = max(r - 3, 0)
        max_row = min(r + 3, self._rows - 1)

        # Horizontal check
        count = 0
        for ci in range(min_col, max_col + 1):
            if self._board[ci][r] == player:
                count += 1
            else:
                count = 0
            if count == 4:
                return True

        # Vertical check
        count = 0
        for ri in range(min_row, max_row + 1):
            if self._board[c][ri] == player:
                count += 1
            else:
                count = 0
            if count == 4:
                return True

        count1 = 0
        count2 = 0
        # Diagonal check
        for i in range(-3, 4):
            # bottom-left -> top-right
            if 0 <= c + i < self._cols and 0 <= r + i < self._rows:
                if self._board[c + i][r + i] == player:
                    count1 += 1
                else:
                    count1 = 0
                if count1 == 4:
                    return True
            # bottom-right -> top-let
            if 0 <= c + i < self._cols and 0 <= r - i < self._rows:
                if self._board[c + i][r - i] == player:
                    count2 += 1
                else:
                    count2 = 0
                if count2 == 4:
                    return True

        # Draw check
        if sum([x.count(0) for x in self._board]) == 0:
            return True

        return False

    def get_cols(self):
        """
        :return: The number of columns of the game
        """
        return self._cols

    def get_rows(self):
        """
        :return: The number of rows of the game
        """
        return self._rows

    def get_win(self):
        """
        :return: If one play won or not
        """
        return self._won

    def get_turn(self):
        """
        :return: To which player is the turn
        """
        return self._turn

    def get_board(self):
        """
        :return: A copy of the game board
        """
        return self._board.copy()

    def board_at(self, c, r):
        """
        :param: c, the column
        :param: r, the row
        :return: What value is held at column c, row r in the board
        """
        return self._board[c][r]

    def copy_state(self):
        """
        Use this instead of the copy() method. Useful as we don't want our graphical interface (viewed as an Observer in this class)
        to be updated when we are playing moves in our tree search.
        """

        # Temporary removes the
        temporary_observers = self._observers
        self._observers = []

        new_one = deepcopy(self)
        new_one._observers.clear()  # Clear observers, such as GUI in our case.

        # Reassign the observers after deepcopy
        self._observers = temporary_observers

        return new_one

    def bot_place(self):
        """
        Calls the bots whenever they need to play their moves
        """
        if self._turn == 1:
            self._player1.make_move()
        else:
            self._player2.make_move()

    def get_valid_locations(self):
        """
        Returns the indices of the columns that are not full, aka the column where
        the user can play his next move
        """
        free_cols = []
        for i in range(COLUMN_COUNT):
            if self._board[i][ROW_COUNT - 1] == 0:
                free_cols.append(i)

        return free_cols


class Connect4Viewer(Observer):
    def __init__(self, game):
        super(Observer, self).__init__()
        assert game is not None
        self._game = game
        self._game.add_observer(self)
        self._screen = None
        self._font = None

    def initialize(self):
        """
        Initialises the view window
        """
        pygame.init()
        pygame.display.set_caption("Connect Four")
        self._font = pygame.font.SysFont(None, 70)
        self._screen = pygame.display.set_mode(
            [
                self._game.get_cols() * SQUARE_SIZE,
                self._game.get_rows() * SQUARE_SIZE,
            ]
        )
        self.draw_board()

    def draw_board(self):
        """
        Draws board[c][r] with c = 0 and r = 0 being bottom left
        0 = empty (background colour)
        1 = yellow
        2 = red
        """
        self._screen.fill(BLUE_COLOR)

        for r in range(self._game.get_rows()):
            for c in range(self._game.get_cols()):
                colour = BACKGROUND_COLOR
                if self._game.board_at(c, r) == 1:
                    colour = YELLOW_COLOR
                if self._game.board_at(c, r) == -1:
                    colour = RED_COLOR

                # Anti-aliased circle drawing
                pygame.gfxdraw.aacircle(
                    self._screen,
                    c * SQUARE_SIZE + SQUARE_SIZE // 2,
                    self._game.get_rows() * SQUARE_SIZE
                    - r * SQUARE_SIZE
                    - SQUARE_SIZE // 2,
                    int(DISC_SIZE_RATIO * SQUARE_SIZE / 2),
                    colour,
                )

                pygame.gfxdraw.filled_circle(
                    self._screen,
                    c * SQUARE_SIZE + SQUARE_SIZE // 2,
                    self._game.get_rows() * SQUARE_SIZE
                    - r * SQUARE_SIZE
                    - SQUARE_SIZE // 2,
                    int(DISC_SIZE_RATIO * SQUARE_SIZE / 2),
                    colour,
                )
        pygame.display.update()

    def update(self, obj, event, *argv):
        """
        Called when notified. Updates the view.
        """
        if event == Event.GAME_WON:
            won = argv[0]
            self.draw_win_message(won)
        elif event == Event.GAME_RESET:
            self.draw_board()
        elif event == Event.PIECE_PLACED:
            self.draw_board()

    def draw_win_message(self, won):
        """
        Displays win message on top of the board
        """
        if won == 1:
            img = self._font.render(
                f"Yellow won ({self._game._player1})",
                True,
                BLACK_COLOR,
                YELLOW_COLOR,
            )
        elif won == -1:
            img = self._font.render(
                f"Red won ({self._game._player2})", True, WHITE_COLOR, RED_COLOR
            )
        else:
            img = self._font.render("Draw", True, WHITE_COLOR, BLUE_COLOR)

        rect = img.get_rect()
        rect.center = (
            (self._game.get_cols() * SQUARE_SIZE) // 2,
            (self._game.get_rows() * SQUARE_SIZE) // 2,
        )

        self._screen.blit(img, rect)
        pygame.display.update()
