"""
based on previous work of : Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import random
import sys

style.use("ggplot")


class Tetris:
    """
    The Tetris environment.
    """

    # RGB colors of the different pieces
    piece_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255),
    ]

    # TEXT CODE FOR THE DIFFERENT ACTIONS
    SIMPLE_MOVEMENT = [
        ["NOOP"],
        ["Rotate"],
        ["Right"],
        ["Left"],
        ["Down"],
    ]

    # Encoding of the various pieces with the different value corresponding to a specific color
    pieces = [
        [[1, 1], [1, 1]],
        [[0, 2, 0], [2, 2, 2]],
        [[0, 3, 3], [3, 3, 0]],
        [[4, 4, 0], [0, 4, 4]],
        [[5, 5, 5, 5]],
        [[0, 0, 6], [6, 6, 6]],
        [[7, 0, 0], [7, 7, 7]],
    ]

    def __init__(self, height=20, width=10, block_size=20):
        """
        Constructor of the Tetris Environment.
        :param height: number of rows for the tetris board
        :param width: number of col for the tetris board
        :param block_size: The size of each tile
        """
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = (
            np.ones(
                (
                    self.height * self.block_size,
                    self.width * int(self.block_size / 2),
                    3,
                ),
                dtype=np.uint8,
            )
            * np.array([204, 204, 255], dtype=np.uint8)
        )
        self.text_color = (200, 20, 220)
        self.reset()

    def reset(self):
        """
        Reset a tetris game to the initial state.
        :return: return the initial state
        """
        self.board = [[0] * self.width for _ in range(self.height)]
        self.tetris_score = 0
        self.tetrominoes = 0
        self.total_cleared_lines = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {
            "x": self.width // 2 - len(self.piece[0]) // 2,
            "y": 0,
        }
        self.landing_height = 0
        self.holes = 0
        self.gameover = False
        return self.get_state_properties(self.board)

    def rotate(self, piece):
        """
        Rotate by 90 degrees the piece selected by a matricial rotation.
        :param piece: Tetromino to be rotated
        :return: the rotated piece
        """
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_state_properties(self, board):
        """
        Returns the different features that will describe the board to the AI.
        Multiples versions have been made but for the final version only 4 features were kept.
        :param board: Actual state of the board
        :return: the various features of the board.
        """
        lines_cleared, board, deleted_rows = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return [lines_cleared, holes, bumpiness, height]

    def get_holes(self, board):
        """
        Get the number of holes. (empty cells which are bounded up by filled cells)
        :param board: current state of the board
        :return: the number of holes
        """
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1 :] if x == 0])
        return num_holes

    def get_cumulative_wells(self, board):
        """
        Returns the sum of all wells.
        :param board: current state of the board.
        :return: sum of wells
        """

        wells = [0 for _ in range(self.width)]

        for y, row in enumerate(board):
            left_empty = False
            for x, code in enumerate(row):
                if code == 0:
                    well = False
                    right_empty = self.width > x + 1 and board[y][x + 1] == 0
                    if left_empty or right_empty:
                        well = True
                    wells[x] = 0 if well else wells[x] + 1

                    left_empty = True
                else:
                    left_empty = False
        return sum(wells)

    def get_column_transitions(self, board):
        """
        Returns the number of vertical cell transitions.
        A cell transition is a passage from a filled cell to empty one or from an empty cell to a filled one
        :param board: current state of the board
        :return: vertical cell transition
        """
        total = 0
        for x in range(self.width):
            column_count = 0
            last_empty = False
            for y in reversed(range(self.height)):
                empty = board[y][x] == 0

                if last_empty and not empty:
                    column_count += 2

                last_empty = empty

            total += column_count

        return total

    def get_row_transitions(self, board):
        """
        Returns the number of horizontal cell transitions.
        :param board: current state of the board
        :return: row cells transition
        """

        total = 0
        for y in range(self.height):
            row_count = 0

            last_fill_state = board[y][0]
            for x in range(1, self.width):
                fill_state = board[y][x] != 0

                if last_fill_state != fill_state:
                    row_count += 1
                    last_fill_state = fill_state

            total += row_count

        return total

    def get_bumpiness_and_height(self, board):
        """
        Return the bumpiness of the board. (The sum of all height difference between each column)
        Return also the height of the highest piece
        :param board:
        :return:
        """
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(
            mask.any(axis=0), np.argmax(mask, axis=0), self.height
        )
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_next_states(self):
        """
        Return all the next future possibles states for the falling piece.
        :return: next possible states
        """
        states = []
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1

                self.landing_height = 20 - pos["y"]
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states.append(((x, i), self.get_state_properties(board)))

            curr_piece = self.rotate(curr_piece)
        return states

    def get_current_board_state(self):
        """
        Get the current state of the board with the falling piece
        :return: the board
        """
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][
                    x + self.current_pos["x"]
                ] = self.piece[y][x]
        return board

    def new_piece(self):
        """
        Place a new tetromino on the board.
        If the tetromino is colliding with another piece then the attribute gameover is set to True.
        :return: None
        """
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {
            "x": self.width // 2 - len(self.piece[0]) // 2,
            "y": 0,
        }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        """
        Check if the piece is colliding with another piece at a certain position.
        A piece is colliding if a part of the piece is touching another piece already on the board.
        :param piece: The piece to check if there is a collision.
        :param pos: The position where we want to verify if there is a collision with the board.
        :return: True if there is a collision, False otherwise.
        """
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if (
                    future_y + y > self.height - 1
                    or self.board[future_y + y][pos["x"] + x]
                    and piece[y][x]
                ):
                    return True
        return False

    def truncate(self, piece, pos):
        """
        Check if there is a condition of the tetris that is not respected and thus will lead to a gameover.
        :param piece: current falling piece
        :param pos: position where the condition must be checked
        :return: True if there is a problem False Otherwise
        """
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if (
            pos["y"] - (len(piece) - last_collision_row) < 0
            and last_collision_row > -1
        ):
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if (
                            self.board[pos["y"] + y][pos["x"] + x]
                            and piece[y][x]
                            and y > last_collision_row
                        ):
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        """
        Make a deep copy of the board with the actual falling piece copied inside the board.
        :param piece: Current Falling Piece
        :param pos:  Position of the piece
        :return: the new Board
        """
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        """
        Check which lines are complete and should be cleared.
        Therefore if a row contains not a single 0 then it should be deleted.
        In a second time, all the full row which are identified are removed from the board.
        :param board: current board
        :return: the board with the removed lines.
        """
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)

        return len(to_delete), board, to_delete

    def remove_row(self, board, indices):
        """
        Removes all rows indicates by the indexes in the parameters. Subsequently, rows full of zeros will replace them.
        :param board: current board
        :param indices: indexes of the row to be deleted
        :return: the board with the deleted row.
        """
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action, render=True, video=None):
        """
        The tetris environment will perform an action and check if the game has ended or not.
        The score of the game with the new action done is computed.
        It is checked if the game is lost.
        Then the score and the binary state of the game is returned.
        :param action: The action to be performed
        :param render: Binary variable which decide if the game should be displaying the move done or not
        :param video: The object which contains all visual components.
        :return: reward received for the move and if the game is over or not.
        """
        x, num_rotations = action
        self.current_pos = {"x": x, "y": 0}
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)

        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            if render:
                self.render(video)
        self.landing_height = 20 - self.current_pos["y"]

        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True

        self.board = self.store(self.piece, self.current_pos)

        score = self.reward()

        return score, self.gameover

    def reward(self):
        """
        Compute the reward of the state of the current board.
        :return: the reward
        """
        lines_cleared, self.board, deleted_rows = self.check_cleared_rows(
            self.board
        )
        self.tetris_score += 1 + (lines_cleared ** 2) * self.width
        self.tetrominoes += 1
        self.total_cleared_lines += lines_cleared

        score = 1 + lines_cleared * 5

        # SET A NEW PIECE IF THE GAME IS NOT OVER.
        if not self.gameover:
            self.new_piece()

        return score

    def tmp_reward(self, board):
        """
        Compute the reward of a state but it does not add a new piece
        which is useful when we want just a check the score.
        :param board: temporary board
        :return: the score of the temp board.
        """
        lines_cleared, board, deleted_rows = self.check_cleared_rows(board)
        score = 1 + lines_cleared * 5

        return score

    def render(self, video=None):
        """
        Build an image of the current state of the game.
        The newly built image is then wrote to a video channel.
        :param video: video channel were the new tetris game image will be written
        :return: None
        """

        if not self.gameover:
            img = [
                self.piece_colors[p]
                for row in self.get_current_board_state()
                for p in row
            ]
        else:
            img = [self.piece_colors[p] for row in self.board for p in row]
        img = (
            np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        )
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = img.resize(
            (self.width * self.block_size, self.height * self.block_size),
            Image.NEAREST,
        )
        img = np.array(img)
        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        img = np.concatenate((img, self.extra_board), axis=1)

        cv2.putText(
            img,
            "Score:",
            (
                self.width * self.block_size + int(self.block_size / 2),
                self.block_size,
            ),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=self.text_color,
        )
        cv2.putText(
            img,
            str(self.tetris_score),
            (
                self.width * self.block_size + int(self.block_size / 2),
                2 * self.block_size,
            ),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=self.text_color,
        )

        cv2.putText(
            img,
            "Pieces:",
            (
                self.width * self.block_size + int(self.block_size / 2),
                4 * self.block_size,
            ),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=self.text_color,
        )
        cv2.putText(
            img,
            str(self.tetrominoes),
            (
                self.width * self.block_size + int(self.block_size / 2),
                5 * self.block_size,
            ),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=self.text_color,
        )

        cv2.putText(
            img,
            "Lines:",
            (
                self.width * self.block_size + int(self.block_size / 2),
                7 * self.block_size,
            ),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=self.text_color,
        )
        cv2.putText(
            img,
            str(self.total_cleared_lines),
            (
                self.width * self.block_size + int(self.block_size / 2),
                8 * self.block_size,
            ),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=self.text_color,
        )

        if video:
            video.write(img)

        cv2.imshow("Deep Q-Learning Tetris", img)

        if cv2.waitKey(1) > 0:
            sys.exit()

        # print(cv2.getWindowProperty("Deep Q-Learning Tetris", 0))
        # print(cv2.getWindowImageRect("Deep Q-Learning Tetris"))

    def setRandomSeed(self, seed):
        random.seed(seed)
