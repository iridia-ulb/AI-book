import pygame, random, time
from pathlib import Path

TITLE = "snAIke!"
FPS = 30
BLACK = (50, 50, 50)
GREY = (120, 120, 120)
WHITE = (200, 200, 200)
YELLOW = (200, 200, 50)
RED = (200, 50, 50)
GREEN = (0, 200, 50)
PURPLE = (200, 0, 150)
ORANGE = (255, 130, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
UP = (-1, 0)
SNAKE_CHAR = "+"
EMPTY_CHAR = " "
WALL_CHAR = "#"
FOOD_CHAR = "@"
NEW_CHAR = "A"
S_CHAR = "S"
CLOSED_CHAR = "C"


class SnakeGame:
    def __init__(self):
        self.run = True
        self.rows = 20
        self.columns = 20
        self.grid = [
            [" " for j in range(self.rows)] for i in range(self.columns)
        ]
        self.snake = []
        self.previous_move = None
        self.next_move = None
        self.food = None
        self.alive = False
        self.score = 0
        self.best_score = 0
        self.start_time = time.time()
        self.current_time = self.start_time
        self.mps = 8
        self.foodEaten = False

    def is_running(self):
        return self.run

    def stop_running(self):
        self.run = False

    def speedup(self):
        if self.mps < 50:
            self.mps += 1
        self.score = 0
        self.best_score = 0

    def slowdown(self):
        if self.mps > 1:
            self.mps -= 1
        self.score = 0
        self.best_score = 0

    def get_mps(self):
        return self.mps

    def reset_grid(self):
        for i in range(20):
            for j in range(20):
                self.grid[i][j] = EMPTY_CHAR
        self.score = 0
        self.best_score = 0

    def expand_row(self):
        if self.rows < 100:
            self.rows += 1
        self.grid = []
        for i in range(self.rows):
            self.grid.append([])
            for j in range(self.columns):
                self.grid[i].append(EMPTY_CHAR)
        self.score = 0
        self.best_score = 0

    def expand_column(self):
        if self.columns < 100:
            self.columns += 1
        self.grid = []
        for i in range(self.rows):
            self.grid.append([])
            for j in range(self.columns):
                self.grid[i].append(EMPTY_CHAR)
        self.score = 0
        self.best_score = 0

    def shrink_row(self):
        if self.rows > 1:
            self.rows -= 1
        self.grid = []
        for i in range(self.rows):
            self.grid.append([])
            for j in range(self.columns):
                self.grid[i].append(EMPTY_CHAR)
        self.score = 0
        self.best_score = 0

    def shrink_column(self):
        if self.columns > 1:
            self.columns -= 1
        self.grid = []
        for i in range(self.rows):
            self.grid.append([])
            for j in range(self.columns):
                self.grid[i].append(EMPTY_CHAR)
        self.score = 0
        self.best_score = 0

    def is_alive(self):
        return self.alive

    def remove_food(self):
        if (
            self.food is not None
            and self.grid[self.food[0]][self.food[1]] == FOOD_CHAR
        ):
            self.grid[self.food[0]][self.food[1]] = EMPTY_CHAR
        self.food = None

    def remove_snake(self):
        for i in range(len(self.snake)):
            pos = self.snake.pop()
            if self.grid[pos[0]][pos[1]] == SNAKE_CHAR:
                self.grid[pos[0]][pos[1]] = EMPTY_CHAR

    def get_available_cells(self):
        available_cells = []
        for i in range(self.rows):
            for j in range(self.columns):
                if self.grid[i][j] == EMPTY_CHAR:
                    available_cells.append((i, j))
        return available_cells

    def get_random_cell(self):
        random_cell = None
        available_cells = self.get_available_cells()
        if len(available_cells) > 0:
            random_cell = random.choice(available_cells)
        return random_cell

    def spawn_snake(self):
        random_cell = self.get_random_cell()
        if random_cell is None:
            self.alive = False
        else:
            self.snake.insert(0, random_cell)
            self.grid[random_cell[0]][random_cell[1]] = SNAKE_CHAR

    def spawn_food(self):
        random_cell = self.get_random_cell()
        if random_cell is None:
            self.alive = False
        else:
            self.grid[random_cell[0]][random_cell[1]] = FOOD_CHAR
            self.food = random_cell

    def start_run(self):
        self.remove_food()
        self.remove_snake()
        self.alive = True
        self.score = 0
        self.previous_move = None
        self.next_move = None
        self.spawn_snake()
        self.spawn_food()
        self.start_time = time.time()
        self.current_time = self.start_time
        # print("Starting game with C={} and R={}".format(self.columns, self.rows))

    def set_next_move(self, move):
        self.next_move = move

    def is_collision(self, pos):
        return not (
            0 <= pos[0] < self.rows
            and 0 <= pos[1] < self.columns
            and self.grid[pos[0]][pos[1]] in [EMPTY_CHAR, FOOD_CHAR]
        )

    def is_next_move_invalid(self):
        if self.previous_move is not None:
            return (
                self.previous_move[0] + self.next_move[0],
                self.previous_move[1] + self.next_move[1],
            ) == (0, 0)

    def move_snake(self):
        if self.next_move == "starve":
            self.alive = False
            if self.score > self.best_score:
                self.best_score = self.score
            return self.get_state()

        if self.next_move is None or self.is_next_move_invalid():
            self.next_move = self.previous_move

        if self.next_move is not None:
            self.foodEaten = False
            head = self.snake[0]
            new_pos = (head[0] + self.next_move[0], head[1] + self.next_move[1])
            if self.is_collision(new_pos):
                self.alive = False
                if self.score > self.best_score:
                    self.best_score = self.score
            else:
                self.snake.insert(0, new_pos)
                self.grid[new_pos[0]][new_pos[1]] = SNAKE_CHAR
                if new_pos == self.food:
                    self.score += 1
                    self.foodEaten = True
                    self.spawn_food()
                else:
                    tail = self.snake.pop()
                    self.grid[tail[0]][tail[1]] = EMPTY_CHAR
                self.previous_move = self.next_move
                self.next_move = None

            return self.get_state()

    def get_state(self):
        return self.grid, self.score, self.alive, self.snake

    def get_grid_base(self, width, height):
        menu_start = width * 2 / 3
        vertical_gap = (height - 1) // self.rows
        horizontal_gap = (menu_start - 1) // self.columns
        gap = min(horizontal_gap, vertical_gap)
        vertical_start = (height - self.rows * gap) // 2
        horizontal_start = (menu_start - self.columns * gap) // 2
        return gap, vertical_start, horizontal_start, menu_start


class GUISnakeGame(SnakeGame):
    DEFAULT_WIDTH = 900
    DEFAULT_HEIGHT = 600
    DEFAULT_TITLE_FONT_SIZE = 40
    DEFAULT_FONT_SIZE = 20

    def __init__(self):
        super(GUISnakeGame, self).__init__()
        self.frame = 0

    def next_tick(self, learning_agent=None):
        self.process_event(learning_agent)
        if self.is_alive() and (
            self.frame / FPS >= 1 / self.get_mps() or learning_agent is not None
        ):
            self.move_snake()
            if self.foodEaten and learning_agent:
                learning_agent.eat()
            self.frame = 0
        # drawing on screen
        self.draw()
        self.clock.tick(FPS)
        self.frame += 1

    def process_event(self, learning_agent=None):
        # triggering an event
        for event in pygame.event.get():
            # closing the game
            if event.type == pygame.QUIT:
                self.stop_running()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.stop_running()
                if not self.is_alive():
                    # start the run
                    if event.key == pygame.K_SPACE:
                        if learning_agent != None:
                            learning_agent.reset_state()
                        self.start_run()

                    # modify speed
                    elif event.key == pygame.K_u:
                        self.slowdown()
                    elif event.key == pygame.K_i:
                        self.speedup()

                    # modify grid
                    elif event.key == pygame.K_r:
                        self.reset_grid()
                    elif event.key == pygame.K_o:
                        self.expand_row()
                    elif event.key == pygame.K_p:
                        self.expand_column()
                    elif event.key == pygame.K_l:
                        self.shrink_row()
                    elif event.key == pygame.K_SEMICOLON:
                        self.shrink_column()

                if self.is_alive():
                    # controls snake
                    if event.key == pygame.K_UP:
                        self.set_next_move(UP)
                    elif event.key == pygame.K_RIGHT:
                        self.set_next_move(RIGHT)
                    elif event.key == pygame.K_DOWN:
                        self.set_next_move(DOWN)
                    elif event.key == pygame.K_LEFT:
                        self.set_next_move(LEFT)

            # resize window
            elif event.type == pygame.VIDEORESIZE:
                self.set_window_size(event.w, event.h)

            if not self.is_alive():
                # left click
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    self.add_wall(pos)
                # right click
                if pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    self.remove(pos)

        if self.is_alive() and learning_agent is not None:
            self.set_next_move(
                learning_agent.choose_next_move(self.get_state())
            )

    def init_pygame(self):
        pygame.init()
        pygame.font.init()

        self.set_window_size(
            GUISnakeGame.DEFAULT_WIDTH, GUISnakeGame.DEFAULT_HEIGHT
        )
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()

    def set_window_size(self, width, height):
        self.screen = pygame.display.set_mode(
            size=(width, height), flags=pygame.RESIZABLE
        )
        ratio = min(
            width / GUISnakeGame.DEFAULT_WIDTH,
            height / GUISnakeGame.DEFAULT_HEIGHT,
        )
        self.title_font = pygame.font.Font(
            Path("Fonts") / Path("Mario-Kart-DS.ttf"),
            round(GUISnakeGame.DEFAULT_TITLE_FONT_SIZE * ratio),
        )
        self.normal_font = pygame.font.Font(
            Path("Fonts") / Path("Fipps-Regular.otf"),
            round(GUISnakeGame.DEFAULT_FONT_SIZE * ratio),
        )

    def get_coord(self, screen, pos):
        gap, vertical_start, horizontal_start, menu_start = self.get_grid_base(
            screen.get_width(), screen.get_height()
        )
        x, y = pos
        i = int((y - vertical_start) // gap)
        j = int((x - horizontal_start) // gap)
        return i, j

    def add_wall(self, pos):
        i, j = self.get_coord(self.screen, pos)
        if 0 <= i < self.rows and 0 <= j < self.columns:
            self.grid[i][j] = WALL_CHAR
        self.score = 0
        self.best_score = 0

    def remove(self, pos):
        i, j = self.get_coord(self.screen, pos)
        if 0 <= i < self.rows and 0 <= j < self.columns:
            self.grid[i][j] = EMPTY_CHAR
        self.score = 0
        self.best_score = 0

    def draw_cells(self, screen, gap, vertical_start, horizontal_start):
        for i in range(self.rows):
            for j in range(self.columns):
                if self.grid[i][j] != EMPTY_CHAR:
                    if self.grid[i][j] == WALL_CHAR:
                        color = WHITE
                    elif self.grid[i][j] == SNAKE_CHAR:
                        color = YELLOW
                    elif self.grid[i][j] == FOOD_CHAR:
                        color = RED
                    elif self.grid[i][j] == NEW_CHAR:
                        color = GREEN
                    elif self.grid[i][j] == S_CHAR:
                        color = PURPLE
                    elif self.grid[i][j] == CLOSED_CHAR:
                        color = ORANGE
                    pygame.draw.rect(
                        screen,
                        color,
                        (
                            horizontal_start + j * gap,
                            vertical_start + i * gap,
                            gap,
                            gap,
                        ),
                    )

    def draw_grid(self, screen, gap, vertical_start, horizontal_start):
        for i in range(self.rows + 1):
            pygame.draw.line(
                screen,
                GREY,
                (horizontal_start, vertical_start + i * gap),
                (
                    horizontal_start + self.columns * gap,
                    vertical_start + i * gap,
                ),
                1,
            )
        for j in range(self.columns + 1):
            pygame.draw.line(
                screen,
                GREY,
                (horizontal_start + j * gap, vertical_start),
                (horizontal_start + j * gap, vertical_start + self.rows * gap),
                1,
            )

    def draw(self):
        self.screen.fill(BLACK)
        width, height = self.screen.get_size()
        gap, vertical_start, horizontal_start, menu_start = self.get_grid_base(
            width, height
        )

        # Draw the map
        self.draw_cells(self.screen, gap, vertical_start, horizontal_start)
        self.draw_grid(self.screen, gap, vertical_start, horizontal_start)
        pygame.draw.line(
            self.screen, GREY, (menu_start, 0), (menu_start, height)
        )

        # Draw texts and timer
        title = self.title_font.render(TITLE, True, WHITE)
        score = self.normal_font.render(
            "Score: " + str(self.score), True, WHITE
        )
        highscore = self.normal_font.render(
            "Highscore: " + str(self.best_score), True, WHITE
        )
        size = self.normal_font.render(
            "Size: " + str(self.rows) + "x" + str(self.columns), True, WHITE
        )
        mps = self.normal_font.render("MPS: " + str(self.mps), True, WHITE)
        start = self.normal_font.render("Press Space", True, WHITE)

        if self.alive:
            self.current_time = time.time()

        timer = self.normal_font.render(
            "Timer: " + str(round(self.current_time - self.start_time, 1)),
            True,
            WHITE,
        )
        self.screen.blit(
            title,
            (
                menu_start + (width - menu_start) / 2 - title.get_width() / 2,
                height * (1 / 15) - title.get_height() / 2,
            ),
        )
        self.screen.blit(
            score,
            (
                menu_start + (width - menu_start) / 7,
                height * (3 / 15) - score.get_height() / 2,
            ),
        )
        self.screen.blit(
            highscore,
            (
                menu_start + (width - menu_start) / 7,
                height * (4 / 15) - highscore.get_height() / 2,
            ),
        )
        self.screen.blit(
            size,
            (
                menu_start + (width - menu_start) / 7,
                height * (5 / 15) - size.get_height() / 2,
            ),
        )
        self.screen.blit(
            mps,
            (
                menu_start + (width - menu_start) / 7,
                height * (6 / 15) - mps.get_height() / 2,
            ),
        )

        if not self.alive:
            self.screen.blit(
                start,
                (
                    menu_start
                    + (width - menu_start) / 2
                    - start.get_width() / 2,
                    height / 2 - start.get_height() / 2,
                ),
            )

        self.screen.blit(
            timer,
            (
                menu_start + (width - menu_start) / 7,
                height - timer.get_height(),
            ),
        )
        pygame.display.flip()

    def cleanup_pygame(self):
        pygame.font.quit()
        pygame.quit()


def display_state_console20x20(state):
    grid, score, alive, head = state
    print("Alive: " + str(alive) + " -- Current reward: " + str(score))

    print("  A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T")
    c = ord("A")
    for line in grid[:20]:
        print(" |" + "-+" * 20)
        print(chr(c) + "|" + "|".join(line[:20]))
        c += 1
