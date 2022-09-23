""" This file runs multiple instances of the Tetris class in synchronization with a PyGame display """
# Imports
import glob
import os
import time
from dataclasses import dataclass
from typing import List
import pygame
import pandas as pd
import random

from TetrisAgents import GeneticAgent
import TetrisUtils as TUtils
from TetrisSettings import *
from Tetris import Tetris


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Parallel Training Settings

# Parallel Tetris game count
ROW_COUNT = 4
COL_COUNT = 6
GAME_COUNT = ROW_COUNT * COL_COUNT

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
class TetrisParallel:
    """
    Class that represents several agents playing individual Tetris games in order to implement a Genetic Algorithm.
    For each generation, all agents play individually until none of them is alive or that the limit time is reached.
    Then, we apply the genetic algorithm and start again with the next generation. The agents of the first generation
    are built randomly. The training stops when the total limit time for the training is reached or when the number
    of generation is reached.

    :param path: Path to the directory where we want to store the details of each generation and the final score
    graph.
    :param nb_ben: Number of generation on which to train (if we do not reach the total limit time before).
    :param limit_time: Time limit per generation.
    :param heuristics_selected: List of the index of HEURISTICS_LABELS that the agent uses. (NB: all agents use the same
    selection of heuristics)
    :param random_run: Boolean telling whether the agents of each generation must be randomly initialized or not.
    :param max_training_time: Total limit time for the entire training.
    NB: this time is computed based on the absolute time from the CPU.
    :param current_gen: Number of the generation we are actually training.
    """

    path: str
    nb_gen: int
    limit_time: int
    heuristics_selected: list
    random_run: bool
    max_training_time: int
    current_gen: int = 1

    # Parameters that will be displayed during the training to keep up
    gen_previous_best_score = 0.0
    gen_top_score = 0.0
    time_elapsed = 0  # Set a time limit so no forever games

    agents = []
    tetris_games = []

    def launch(self):
        """
        Function that launches the training of the agents: initialize the pygame and while the agents are not dead
        or the time limit is not reached, asks for next step and updates the screen. Display each generation after
        the other, until the limit of generations or total time is reached.
        """
        print(
            f">> Initializing {GAME_COUNT} Tetris games in parallel with a grid of {ROW_COUNT}×{COL_COUNT}..."
        )
        print(f"The heuristics selected were : {self.heuristics_selected}")

        # Initialize PyGame module
        pygame.init()
        pygame.font.init()
        display_screen = pygame.display.set_mode(size=(SCREEN_WIDTH, SCREEN_HEIGHT))
        print(f">> Screen size calculated to {SCREEN_WIDTH}×{SCREEN_HEIGHT}...")

        # Initialize Tetris modules and agents
        start_time = time.process_time()
        print(f">> Initializing {GAME_COUNT} Tetris agent(s)...")
        for _ in range(GAME_COUNT):
            self.tetris_games.append(Tetris())
            self.agents.append(GeneticAgent(self.heuristics_selected))

        print(f">> Initializing the save directory...")
        # Clearing the existing model_gen files in the given path
        if os.path.exists(self.path):
            files = glob.glob(self.path + "/model_gen_*.csv")
            for f in files:
                os.remove(f)
        else:
            os.makedirs(self.path)

        print(f">> Initialization complete! Let the show begin!")
        running = True
        end_time = 0
        while self.current_gen <= self.nb_gen and running and end_time - start_time <= self.max_training_time:
            # Each loop iteration is 1 frame
            event = self.update(display_screen)
            for e in event:
                if e.type == pygame.QUIT:
                    running = False
            end_time = time.process_time()
        print(f"Training time : {end_time-start_time}")

    def update(self, screen):
        """Called every frame by the runner, handles updates each frame"""
        self.time_elapsed += 1

        # Check if all agents have reached game over state
        if all(tetris.game_over for tetris in self.tetris_games) or (
            self.limit_time != -1 and self.time_elapsed % self.limit_time == 0
        ):
            # Save the generation agents information
            df = save_gen(self.agents, self.tetris_games)
            df.to_csv(f"{self.path}/model_gen_{self.current_gen}.csv", encoding="utf-8", index=False)

            # Update the generation
            self.time_elapsed = 0
            # Everyone "died" or time's up, select best one and cross over
            combos = zip(self.agents, self.tetris_games)
            parents = sorted(combos, key=lambda combo: combo[1].score, reverse=True)
            # Update generation information
            self.current_gen += 1
            self.gen_previous_best_score = parents[0][1].score
            if self.gen_previous_best_score > self.gen_top_score:
                self.gen_top_score = self.gen_previous_best_score
            # Undo zipping
            parents = [a[0] for a in parents]
            # Discard 50% of population
            parents = parents[: GAME_COUNT // 2]
            # Keep first place agent
            self.agents = [parents[0]]
            if self.random_run:
                self.agents = []

            # Randomly breed the rest of the agents
            while len(self.agents) < GAME_COUNT:
                parent1, parent2 = random.sample(parents, 2)
                if self.random_run:
                    self.agents.append(GeneticAgent(self.heuristics_selected))
                else:
                    self.agents.append(parent1.breed(parent2))

            # Reset games
            for tetris in self.tetris_games:
                tetris.reset_game()

        for a in range(GAME_COUNT):
            # If game over, ignore
            if self.tetris_games[a].game_over:
                continue
            self.tetris_games[a].step(self.agents[a].get_action(self.tetris_games[a]))

        self.draw(screen)
        return pygame.event.get()

    def draw(self, screen):
        """Called by the update() function every frame, draws the PyGame GUI"""
        # Background layer
        screen.fill(TUtils.get_color_tuple(COLORS.get("BACKGROUND_BLACK")))

        # Draw Tetris boards
        curr_x, curr_y = PADDING, PADDING
        for x in range(ROW_COUNT):
            for y in range(COL_COUNT):
                self.draw_board(screen, self.tetris_games[x * COL_COUNT + y], curr_x, curr_y)
                curr_x += GAME_WIDTH + PADDING
            curr_x = PADDING
            curr_y += GAME_HEIGHT + PADDING

        # Draw statistics
        # Realign starting point to statistics bar
        curr_x, curr_y = GAME_WIDTH * COL_COUNT + PADDING * (COL_COUNT + 1), PADDING

        # Draw title
        self.draw_text("Tetris", screen, (curr_x, curr_y), font_size=48)
        curr_y += 60
        # Draw statistics
        best_indexes, best_score = self.get_high_score()
        self.draw_text(f"High Score: {best_score:.1f}", screen, (curr_x, curr_y))
        curr_y += 20
        self.draw_text(f"Best Agent: {SEP.join(map(str, best_indexes))}", screen, (curr_x, curr_y))
        curr_y += 20

        # Draw genetics
        if self.current_gen > -1:
            curr_y += 20
            self.draw_text(f"Generation #{self.current_gen}", screen, (curr_x, curr_y), font_size=24)
            curr_y += 35
            self.draw_text(f"Time Limit: {self.time_elapsed}/{self.limit_time}", screen, (curr_x, curr_y))
            curr_y += 20

            survivor = len([a for a in self.tetris_games if not a.game_over])
            self.draw_text(
                f"Survivors: {survivor}/{GAME_COUNT} ({survivor / GAME_COUNT * 100:.1f}%)",
                screen,
                (curr_x, curr_y),
            )
            curr_y += 20
            self.draw_text(f"Prev H.Score: {self.gen_previous_best_score:.1f}", screen, (curr_x, curr_y))
            curr_y += 20
            self.draw_text(f"All Time H.S: {self.gen_top_score:.1f}", screen, (curr_x, curr_y))
            curr_y += 40

            # Display selected agent
            mouse_x, mouse_y = pygame.mouse.get_pos()
            grid_x, grid_y = mouse_x // (GAME_WIDTH + PADDING), mouse_y // (GAME_HEIGHT + PADDING)
            selected = grid_y * COL_COUNT + grid_x

            agent_index = -1
            highlight_selected = False
            if selected < GAME_COUNT and mouse_x < (GAME_WIDTH + PADDING) * COL_COUNT:
                agent_index = selected
                highlight_selected = True
            elif len(best_indexes) > 0:
                agent_index = best_indexes[0]

            if agent_index != -1:
                self.draw_text(f"Agent #{agent_index}:", screen, (curr_x, curr_y), font_size=24)
                curr_y += 35

                for index in self.agents[agent_index].weight_to_consider:
                    self.draw_text(
                        f">> {HEURISTICS_LABELS[index]}: {self.agents[agent_index].weight_array[index]:.3f}",
                        screen,
                        (curr_x, curr_y),
                    )
                    curr_y += 20

                if highlight_selected:
                    self.highlight(screen, selected, mode=1)
                else:
                    # Highlight current best(s)
                    for a in best_indexes:
                        self.highlight(screen, a, mode=0)

            # Update display
            pygame.display.update()

    def highlight(self, screen, index: int, mode: int):
        """
        Highlight a certain Tetris grid

        :param screen: the screen to draw on
        :param index: index of Tetris grid to highlight
        :param mode: 0/1, 0 = best, 1 = previous best
        """
        game_x = index % COL_COUNT
        game_y = index // COL_COUNT
        color = TUtils.get_color_tuple(COLORS.get("HIGHLIGHT_GREEN" if mode == 0 else "HIGHLIGHT_RED"))

        if mode == 1:
            # Draw previous best (thick border)
            temp_x = (GAME_WIDTH + PADDING) * game_x
            temp_y = (GAME_HEIGHT + PADDING) * game_y
            pygame.draw.rect(screen, color, (temp_x, temp_y, GAME_WIDTH + PADDING * 2, PADDING))
            pygame.draw.rect(screen, color, (temp_x, temp_y, PADDING, GAME_HEIGHT + PADDING * 2))
            temp_x = (GAME_WIDTH + PADDING) * (game_x + 1) + PADDING - 1
            temp_y = (GAME_HEIGHT + PADDING) * (game_y + 1) + PADDING - 1
            pygame.draw.rect(screen, color, (temp_x, temp_y, -GAME_WIDTH - PADDING, -PADDING))
            pygame.draw.rect(screen, color, (temp_x, temp_y, -PADDING, -GAME_HEIGHT - PADDING))
        elif mode == 0:
            # Draw current best (thin border)
            temp_x = (GAME_WIDTH + PADDING) * game_x + PADDING / 2
            temp_y = (GAME_HEIGHT + PADDING) * game_y + PADDING / 2
            pygame.draw.rect(screen, color, (temp_x, temp_y, GAME_WIDTH + PADDING, PADDING / 2))
            pygame.draw.rect(screen, color, (temp_x, temp_y, PADDING / 2, GAME_HEIGHT + PADDING))
            temp_x = (GAME_WIDTH + PADDING) * (game_x + 1) + PADDING / 2 - 1
            temp_y = (GAME_HEIGHT + PADDING) * (game_y + 1) + PADDING / 2 - 1
            pygame.draw.rect(screen, color, (temp_x, temp_y, -GAME_WIDTH - PADDING + 2, -PADDING / 2))
            pygame.draw.rect(screen, color, (temp_x, temp_y, -PADDING / 2, -GAME_HEIGHT - PADDING + 2))

    def get_high_score(self):
        best_indexes, best_score = [], 0
        for a in range(GAME_COUNT):
            # Ignore dead games
            if self.tetris_games[a].game_over:
                continue
            # Get score
            score = self.tetris_games[a].score
            if score > best_score:
                best_indexes = [a]
                best_score = score
            elif score == best_score:
                best_indexes.append(a)
        return best_indexes, best_score

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


"""
Save the information of a generation into a csv file
"""
def get_agent_dict(agent : GeneticAgent):
    """
    Get the dict of the agent's weight
    """
    data = {
        "weight_holes" : round(agent.weight_array[0], 3),
        "weight_height" : round(agent.weight_array[1],3),
        "weight_bumpiness" : round(agent.weight_array[2],3),
        "weight_line_clear" : round(agent.weight_array[3],3),
        "weight_hollow_columns" : round(agent.weight_array[4], 3),
        "weight_row_transition" : round(agent.weight_array[5],3),
        "weight_col_transition": round(agent.weight_array[6],3),
        "weight_pit_count" : round(agent.weight_array[7],3),
        "weight_to_consider " : agent.weight_to_consider,
    }
    return data

def save_gen(agents_list : List[GeneticAgent], tetris_games : List[Tetris]):
    """
    Save a generation of the tetris game into a csv file.
    """
    data=[]
    for agent_index in range(len(agents_list)):
        agent_dict = get_agent_dict(agents_list[agent_index])
        agent_dict["score"] = tetris_games[agent_index].score
        data.append(agent_dict)

    gen_df = pd.DataFrame(data)
    return gen_df
