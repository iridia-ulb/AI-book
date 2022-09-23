# Configurations (USER)
SIZE_SCALE = 1

FONT_NAME = "Consolas"

# Color theme referenced from:
# https://www.mathsisfun.com/games/tetris.html
COLORS = {
    # Display
    "BACKGROUND_BLACK": "000000",
    "BACKGROUND_DARK": "021c2d",
    "BACKGROUND_LIGHT": "00263f",
    "TRIANGLE_GRAY": "efe6ff",
    "WHITE": "ffffff",
    "RED": "ff0000",
    # Tetris pieces
    "TILE_LINE": "ffb900",
    "TILE_L": "2753f1",
    "TILE_L_REVERSED": "f7ff00",
    "TILE_S": "ff6728",
    "TILE_S_REVERSED": "11c5bf",
    "TILE_T": "ae81ff",
    "TILE_CUBE": "e94659",
    # Highlights
    "HIGHLIGHT_GREEN": "22ee22",
    "HIGHLIGHT_RED": "ee2222",
}

SEP = ", "

# Configurations (SYSTEM)
GRID_ROW_COUNT = 20
GRID_COL_COUNT = 10

SCREEN_RATIO = 0.55
SCREEN_WIDTH = int(360 / SCREEN_RATIO * SIZE_SCALE)
SCREEN_HEIGHT = int(720 * SIZE_SCALE)

########################
# Score Configurations #
########################
MULTI_SCORE_ALGORITHM = lambda lines_cleared: ((2 ** lines_cleared) * GRID_COL_COUNT)

######################
# STEP Configuration #
######################

ACTIONS = ["NOTHING", "L", "R", "2L", "2R", "ROTATE", "SWAP", "FAST_FALL", "INSTA_FALL"]

######################
# Tile Configuration #
######################
TILES = ["LINE", "L", "L_REVERSED", "S", "S_REVERSED", "T", "CUBE"]
TILE_SHAPES = {
    "LINE": [[1, 1, 1, 1]],
    "L": [[0, 0, 2],
          [2, 2, 2]],
    "L_REVERSED": [[3, 0, 0],
                   [3, 3, 3]],
    "S": [[0, 4, 4],
          [4, 4, 0]],
    "S_REVERSED": [[5, 5, 0],
                   [0, 5, 5]],
    "T": [[6, 6, 6],
          [0, 6, 0]],
    "CUBE": [[7, 7],
             [7, 7]]
}
