# Imports
from dataclasses import dataclass
import pygame
import time

from TetrisAgents import GeneticAgent
from Tetris import Tetris
import TetrisUtils as TUtils
from TetrisSettings import *

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Parallel Training Settings

# We only have one agent (keep the ROW and COL_COUNT for ease)
ROW_COUNT = 1
COL_COUNT = 1

# Size of each Tetris display
GAME_WIDTH = 120
GAME_HEIGHT = GAME_WIDTH * 2  # no need to modify
GAME_GRID_SIZE = GAME_WIDTH / GRID_COL_COUNT  # no need to modify

# Size of padding
PADDING = 10
PADDING_STATS = 300

# Screen size (automatically calculated)
SCREEN_WIDTH = GAME_WIDTH * COL_COUNT + PADDING * (COL_COUNT + 1) + PADDING_STATS
SCREEN_HEIGHT = GAME_HEIGHT * ROW_COUNT + PADDING * (ROW_COUNT + 1)

# Mutation Rate
MUTATION_RATE = 0.1  # 10% mutation chance

# End of Settings
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# List of heuristics labels to display
HEURISTICS_LABELS = [
    "Hole Count",
    "Agg Height",
    "Bumpiness",
    "Line Clear",
    "Hollow Column",
    "Row Transition",
    "Column Transition",
    "Pit Count",
]


@dataclass
class TetrisSolo:
    """
    Class that represent one single Tetris game, initialized with a specific agent.
    NB: This class is only used for evaluating the agent, never to train it (as a GA would require several agents).

    :param tetrominoes_limit: the maximum number of tetrominoes that the agent will receive for its evaluations. It could
        receive less if it dies before reaching the limit.
    :param heuristic_select: a list of the index of HEURISTICS_LABELS that the agent uses.
    :param agent: a Tetris agent with its own weights for each heuristic that it considers.
    :param tetris_game: a new instance of Tetris such that the agent begins its evaluation from the start.
    """

    tetrominoes_limit: int
    heuristics_selected: list
    agent: GeneticAgent
    tetris_game = Tetris()

    def launch(self):
        """
        Function that launches the evaluation of the agent: initialize the pygame and while the agent is not dead
        or the limit number of tetrominoes is not reached, asks for next step and updates the screen.
        """
        self.tetris_game = Tetris()
        print(f">> Initializing the best Tetris game...")
        print(f"The heuristics selected were : {self.heuristics_selected}")

        # Initialize PyGame module and music
        pygame.init()
        pygame.font.init()
        # pygame.mixer.music.load("OST/TetrisTheme.mp3")
        # pygame.mixer.music.play()
        display_screen = pygame.display.set_mode(size=(SCREEN_WIDTH, SCREEN_HEIGHT))

        # Initialize Tetris modules and agents
        print(f">> Initialization complete! Let the show begin!")

        running = True
        while (
            not self.tetris_game.game_over
            and self.tetris_game.tetrominoes_number <= self.tetrominoes_limit
            and running
        ):
            # NB: can comment it such that the evaluation is done faster. Better with it for vizualisation
            time.sleep(0.01)
            # Each loop iteration is 1 frame
            event = self.update(display_screen)
            for e in event:
                if e.type == pygame.QUIT:
                    running = False

        print(f">> Finito ! Score reached: {self.tetris_game.score}")
        pygame.display.quit()
        pygame.quit()

    def update(self, screen):
        self.tetris_game.step(self.agent.get_action(self.tetris_game))
        self.draw(screen)
        return pygame.event.get()

    def draw(self, screen):
        """Called by the update() function every frame, draws the PyGame GUI"""
        # Background layer
        screen.fill(TUtils.get_color_tuple(COLORS.get("BACKGROUND_BLACK")))

        # Draw Tetris boards
        curr_x, curr_y = PADDING, PADDING
        self.draw_board(screen, self.tetris_game, curr_x, curr_y)

        # Realign starting point to statistics bar
        curr_x, curr_y = GAME_WIDTH * COL_COUNT + PADDING * (COL_COUNT + 1), PADDING

        # Draw title
        self.draw_text("Tetris", screen, (curr_x, curr_y), font_size=48)
        curr_y += 60
        # Draw statistics
        self.draw_text(f"Score: {self.tetris_game.score:.1f}", screen, (curr_x, curr_y))
        curr_y += 20
        self.draw_text(
            f"Number of tetrominoes: {self.tetris_game.tetrominoes_number:.1f}", screen, (curr_x, curr_y)
        )
        pygame.display.update()

    def draw_text(self, message: str, screen, offsets, font_size=16, color="WHITE"):
        """Draws a line of text at the specified offsets"""
        text_image = pygame.font.SysFont(FONT_NAME, font_size).render(
            message, False, TUtils.get_color_tuple(COLORS.get(color))
        )
        screen.blit(text_image, offsets)

    def draw_board(self, screen, tetris: Tetris, x_offset: int, y_offset: int):
        """
        Draws one Tetris board with offsets, called by draw() multiple times per frame

        :param screen: the screen to draw on
        :param tetris: Tetris instance
        :param x_offset: X offset (starting X)
        :param y_offset: Y offset (starting Y)
        """
        # [0] Striped background layer
        for a in range(GRID_COL_COUNT):
            color = TUtils.get_color_tuple(COLORS.get("BACKGROUND_DARK" if a % 2 == 0 else "BACKGROUND_LIGHT"))
            pygame.draw.rect(
                screen, color, (x_offset + a * GAME_GRID_SIZE, y_offset, GAME_GRID_SIZE, GAME_HEIGHT)
            )

        # [1] Board tiles
        self.draw_tiles(screen, tetris.board, global_offsets=(x_offset, y_offset))
        # [1] Current tile
        self.draw_tiles(
            screen,
            tetris.tile_shape,
            offsets=(tetris.tile_x, tetris.tile_y),
            global_offsets=(x_offset, y_offset),
        )

        # [2] Game over graphics
        if tetris.game_over:
            color = TUtils.get_color_tuple(COLORS.get("BACKGROUND_BLACK"))
            ratio = 0.9
            pygame.draw.rect(
                screen,
                color,
                (x_offset, y_offset + (GAME_HEIGHT * ratio) / 2, GAME_WIDTH, GAME_HEIGHT * (1 - ratio)),
            )

            message = "GAME OVER"
            color = TUtils.get_color_tuple(COLORS.get("RED"))
            text_image = pygame.font.SysFont(FONT_NAME, GAME_WIDTH // 6).render(message, False, color)
            rect = text_image.get_rect()
            screen.blit(
                text_image,
                (x_offset + (GAME_WIDTH - rect.width) / 2, y_offset + (GAME_HEIGHT - rect.height) / 2),
            )

    def draw_tiles(self, screen, matrix, offsets=(0, 0), global_offsets=(0, 0), outline_only=False):
        """
        Draw tiles from a matrix (utility method)

        :param screen: the screen to draw on
        :param matrix: the matrix to draw
        :param offsets: matrix index offsets
        :param global_offsets: global pixel offsets
        :param outline_only: draw prediction outline only?
        """
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val == 0:
                    continue
                coord_x = global_offsets[0] + (offsets[0] + x) * GAME_GRID_SIZE
                coord_y = global_offsets[1] + (offsets[1] + y) * GAME_GRID_SIZE
                # Draw rectangle
                if not outline_only:
                    pygame.draw.rect(
                        screen,
                        TUtils.get_color_tuple(COLORS.get("TILE_" + TILES[val - 1])),
                        (coord_x, coord_y, GAME_GRID_SIZE, GAME_GRID_SIZE),
                    )
                    pygame.draw.rect(
                        screen,
                        TUtils.get_color_tuple(COLORS.get("BACKGROUND_BLACK")),
                        (coord_x, coord_y, GAME_GRID_SIZE, GAME_GRID_SIZE),
                        1,
                    )
                    # Draw highlight triangle
                    offset = int(GAME_GRID_SIZE / 10)
                    pygame.draw.polygon(
                        screen,
                        TUtils.get_color_tuple(COLORS.get("TRIANGLE_GRAY")),
                        (
                            (coord_x + offset, coord_y + offset),
                            (coord_x + 3 * offset, coord_y + offset),
                            (coord_x + offset, coord_y + 3 * offset),
                        ),
                    )
                else:
                    # Outline-only for prediction location
                    pygame.draw.rect(
                        screen,
                        TUtils.get_color_tuple(COLORS.get("TILE_" + TILES[val - 1])),
                        (coord_x + 1, coord_y + 1, GAME_GRID_SIZE - 2, GAME_GRID_SIZE - 2),
                        1,
                    )

