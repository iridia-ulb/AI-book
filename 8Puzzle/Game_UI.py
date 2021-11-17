import sys, os, random
import re
import pathlib
import pygame

# Constant for the size of the screen.
WIDTH = 800
HEIGHT = 500
# Constant for the size of the text.
TXT_CORE_SIZE = 38
TXT_MENU_SIZE = 50


class SlidePuzzle:
    def __init__(self, gs, ts, ms, screen):
        """
        Init the game.

        :param gs: The grid size. It is a tuple (n,n) of Int.
        :param ts: The size of the tiles. It is an Int.
        :param ms: The size of the margin. It is an Int.
        :param screen:      The screen, Surface object.
        """
        # Attributes for the main core of the game.
        self.screen = screen

        self.gs, self.ts, self.ms = gs, ts, ms
        self.tiles_len = gs[0] * gs[1] - 1
        # Tiles is a list of position [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
        # for gs = (3, 3).
        self.tiles = [(x, y) for y in range(gs[1]) for x in range(gs[0])]
        # The win condition is the same list but we do not want to change to compare it with tiles.
        self.winCdt = [(x, y) for y in range(gs[1]) for x in range(gs[0])]

        # actual pos on the screen.
        self.tilepos = [
            (x * (ts + ms) + ms, y * (ts + ms) + ms)
            for y in range(gs[1])
            for x in range(gs[0])
        ]

        # the place they slide to.
        self.tilePOS = {
            (x, y): (x * (ts + ms) + ms, y * (ts + ms) + ms)
            for y in range(gs[1])
            for x in range(gs[0])
        }
        self.nb_move = 0
        # Speed for the move of tiles.
        self.speed = 3
        # Previous tile.
        self.prev = None
        # Boolean if the player want to return to the main menu.
        self.want_to_quit = False
        # boolean for toogling autoPlay
        self.autoPlay = False
        self.last = 0

        # Attributes for the image of the game.
        # Create a rectangle for the game according to the grid size, the size of tiles and the size of
        # margin.
        self.rect = pygame.Rect(
            0, 0, gs[0] * (ts + ms) + ms, gs[1] * (ts + ms) + ms
        )
        self.pic = pygame.transform.smoothscale(
            pygame.image.load(
                pathlib.Path("assets") / pathlib.Path("image.png")
            ),
            self.rect.size,
        )
        # Partition the image according to the number of tiles.
        self.images = []
        font = pygame.font.Font(None, 120)
        for i in range(self.tiles_len):
            x, y = self.tilepos[i]
            image = self.pic.subsurface(x, y, ts, ts)
            text = font.render(str(i + 1), 2, (0, 0, 0))
            w, h = text.get_size()
            image.blit(text, ((ts - w) / 2, (ts - h) / 2))
            self.images += [image]

    def drawBar(self, pos, size, borderC, barC, progress):
        """
        Draw the loading bar screen.

        :param pos:         2-Tuple of integers corresponding to the (x,y) position of the bar.
        :param size:        2-Tuple of integers corresponding to the size of the bar.
        :param borderC:     3-Tuple of integers corresponding to the RGB color of bar's border.
        :param barC:        3-Tuple of integers corresponding to the RGB color of the bar.
        :param progress:    A double corresponding to the percentage of the progression.
        """
        self.screen.fill((0, 0, 0))
        self.drawText(
            "The AI is pre-training, please wait ...",
            TXT_MENU_SIZE,
            400,
            130,
            255,
            255,
            255,
            True,
        )
        pygame.draw.rect(self.screen, borderC, (*pos, *size), 1)
        innerPos = (pos[0] + 3, pos[1] + 3)
        innerSize = ((size[0] - 6) * progress, size[1] - 6)
        pygame.draw.rect(self.screen, barC, (*innerPos, *innerSize))
        pygame.display.flip()

    def getBlank(self):
        """
        Get the blank tile, the empty tile.

        :return: Return the last tile, the position of the last tile. It is a tuple of Int (x, y).
        """
        return self.tiles[-1]

    def setBlank(self, pos):
        """
        Set the blank tile, the empty tile.

        :param pos: The position of the blank tile. It is a tuple of Int (x, y).
        """
        self.tiles[-1] = pos

    # The blank tile, the empty tile.
    opentile = property(getBlank, setBlank)

    def isWin(self):
        """
        Check if the game is won.

        :return: Return a Boolean, it is True if the game is won, otherwise False.
        """
        if self.tiles == self.winCdt:
            return True
        return False

    def sliding(self):
        """
        Check if there is a tile that is sliding.

        :return: Return a Boolean, True if a tile is sliding, otherwise None.
        """
        for i in range(self.tiles_len):
            x, y = self.tilepos[i]  # current pos
            X, Y = self.tilePOS[self.tiles[i]]  # target pos
            if x != X or y != Y:
                return True

    def switch(self, tile, isAIPlaying):
        """
        Switch the current tile with the blank, where a tile is a tuple (x, y) of int.

        :param tile:        The current tile, a tuple (x, y) of Int.
        :param isAIPlaying: Boolean, True if the AI is currently playing, False otherwise.
        :return:            Break the switch function if a tile is sliding, None value.
        """
        # Since we can keep moving tiles while others are sliding, we should stop that from happening.
        # We attempt this using the sliding function.
        if not isAIPlaying:
            if self.sliding():
                return
        self.tiles[self.tiles.index(tile)], self.opentile, self.prev = (
            self.opentile,
            tile,
            self.opentile,
        )
        self.nb_move += 1

    def inGrid(self, tile):
        """
        Check if the tile is in the grid.

        :param tile: The tile to check, a tuple (x, y) of int.
        :return:     Return a Boolean, it is True if the tile is in the grid, otherwise False.
        """
        return (
            tile[0] >= 0
            and tile[0] < self.gs[0]
            and tile[1] >= 0
            and tile[1] < self.gs[1]
        )

    def adjacent(self):
        """
        Give the positions of the tiles adjacent to the blank tile.

        :return: Return positions of the tiles adjacent to the blank tile, 4 tuples (x, y) of Int.
        """
        x, y = self.opentile
        return (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)

    def random(self):
        """
        Choose randomly an action.
        """
        adj = self.adjacent()
        self.switch(
            random.choice(
                [pos for pos in adj if self.inGrid(pos) and pos != self.prev]
            ),
            False,
        )

    def shuffle(self):
        """
        Shuffle tiles and check if the board is solvable.
        """
        print("shuf")
        while not self.isSolvable():
            random.shuffle(self.tiles)

    def isSolvable(self):
        """
        Check if the game is solvable.

        :return: Return a Boolean, True if the board is solvable, otherwise False.
        """
        tiles = []
        for i in range(len(self.tiles)):
            for j in range(len(self.tiles)):
                if self.tiles[j][1] * 3 + self.tiles[j][0] + 1 == i + 1:
                    tiles.append(j + 1)
        count = 0
        for i in range(len(tiles) - 1):
            for j in range(i + 1, len(tiles)):
                if tiles[i] > tiles[j] and tiles[i] != 9:
                    count += 1
        return True if (count % 2 == 0 and count != 0) else False

    def update(self, dt):
        """
        Update the view.

        :param dt: Derived time. It is an Int.
        """
        # If the value between the current and target is less than speed, we can just let it jump right into place.
        # Otherwise, we just need to add/sub in direction.
        s = self.speed * dt
        for i in range(self.tiles_len):
            x, y = self.tilepos[i]  # current pos
            X, Y = self.tilePOS[self.tiles[i]]  # target pos
            dx, dy = X - x, Y - y

            self.tilepos[i] = (
                X if abs(dx) < s else x + s if dx > 0 else x - s
            ), (Y if abs(dy) < s else y + s if dy > 0 else y - s)

    def draw(self):
        """
        Draw the game with the number of move.
        """
        for i in range(self.tiles_len):
            x, y = self.tilepos[i]
            self.screen.blit(self.images[i], (x, y))
        self.drawText(
            "Moves : {}".format(self.nb_move),
            TXT_CORE_SIZE,
            500,
            10,
            255,
            255,
            255,
            False,
        )

    def drawText(self, text, size, x, y, R, G, B, center):
        """
        Draw text.

        :param text:    The text to draw on the  String.
        :param size:    The size of the text, Int.
        :param x:       The x position of the text, Int.
        :param y:       The y position of the text, Int.
        :param R:       The R color, Int.
        :param G:       The G color, Int.
        :param B:       The B color, Int.
        :param center:  If the text need to be in the center, Boolean.
        """
        font = pygame.font.Font(None, size)
        text = font.render(text, True, (R, G, B))
        if center:
            text_rect = text.get_rect()
            text_rect.midtop = (x, y)
            self.screen.blit(text, text_rect)
        else:
            self.screen.blit(text, (x, y))

    def drawShortcuts(self, is_player, qValues):
        """
        Draw in game shortcuts.

        :param is_player:   A Boolean, it checks if it is a player because, shorcuts are different in
                            the player mode or in the AI mode.
        """
        self.drawText("Shortcuts", TXT_CORE_SIZE, 500, 40, 255, 255, 255, False)
        self.drawText(
            "Pause: Escape", TXT_CORE_SIZE, 500, 70, 255, 255, 255, False
        )
        if is_player:
            self.drawText(
                "Move up: <z>", TXT_CORE_SIZE, 500, 100, 255, 255, 255, False
            )
            self.drawText(
                "Move down: <s>", TXT_CORE_SIZE, 500, 130, 255, 255, 255, False
            )
            self.drawText(
                "Move left: <q>", TXT_CORE_SIZE, 500, 160, 255, 255, 255, False
            )
            self.drawText(
                "Move right: <d>",
                TXT_CORE_SIZE,
                500,
                190,
                255,
                255,
                255,
                False,
            )
            self.drawText(
                "Random move: <Space>",
                TXT_CORE_SIZE,
                500,
                220,
                255,
                255,
                255,
                False,
            )
        else:
            self.drawText(
                "AI move: <Space>",
                TXT_CORE_SIZE,
                500,
                100,
                255,
                255,
                255,
                False,
            )
            self.drawText(
                "auto AI move: <a>",
                TXT_CORE_SIZE,
                500,
                130,
                255,
                255,
                255,
                False,
            )
            if qValues != None:
                self.drawText(
                    "State's Q-Values",
                    TXT_CORE_SIZE,
                    500,
                    190,
                    255,
                    255,
                    255,
                    False,
                )
                self.drawText(
                    "UP: {0:.2E}".format(qValues[0]),
                    TXT_CORE_SIZE,
                    500,
                    220,
                    255,
                    255,
                    255,
                    False,
                )
                self.drawText(
                    "RIGHT: {0:.2E}".format(qValues[1]),
                    TXT_CORE_SIZE,
                    500,
                    250,
                    255,
                    255,
                    255,
                    False,
                )
                self.drawText(
                    "DOWN: {0:.2E}".format(qValues[2]),
                    TXT_CORE_SIZE,
                    500,
                    280,
                    255,
                    255,
                    255,
                    False,
                )
                self.drawText(
                    "LEFT: {0:.2E}".format(qValues[3]),
                    TXT_CORE_SIZE,
                    500,
                    310,
                    255,
                    255,
                    255,
                    False,
                )

    def playEvents(self, event):
        """
        Catch events from the mouse and the keyboard.
        Binded keys:
            - z moves the tile upwards
            - s moves the tile downwards
            - q moves the tile to the left
            - d moves the tile to the right
            - space choose a random mouvement

        :param event:   The current event, Event object.
        """
        mouse = pygame.mouse.get_pressed()
        mpos = pygame.mouse.get_pos()
        # If we use the left click
        if mouse[0]:
            # We convert the position of the mouse according to the grid position and the margin
            x, y = mpos[0] % (self.ts + self.ms), mpos[1] % (self.ts + self.ms)
            if x > self.ms and y > self.ms:
                tile = mpos[0] // self.ts, mpos[1] // self.ts
                if self.inGrid(tile) and tile in self.adjacent():
                    self.switch(tile, False)

        if event.type == pygame.KEYDOWN:
            for key, dx, dy in (
                (pygame.K_s, 0, -1),
                (pygame.K_z, 0, 1),
                (pygame.K_d, -1, 0),
                (pygame.K_q, 1, 0),
            ):
                if event.key == key:
                    x, y = self.opentile
                    tile = x + dx, y + dy
                    if self.inGrid(tile):
                        self.switch(tile, False)
            # Move randomly a tile.
            if event.key == pygame.K_SPACE:
                for _ in range(1000):
                    self.random()

    def catchExitEvent(self, event):
        """
        Check if it is a quit event.

        :param event:   Event object.
        """
        if event.type == pygame.QUIT:
            self.exit()

    def catchGameEvents(self, is_player, movefunc):
        """
        Catchs event during the game and during the board creation.

        :param is_player:       A boolean value to check if it is a game with a player or with an AI.
        :return:                Return True if the player want to quit the game.
                                Otherwise, False.
        """
        for event in pygame.event.get():
            self.catchExitEvent(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.want_to_quit = self.pauseMenu()
                    return
            if is_player:
                self.playEvents(event)
            elif (
                not is_player
                and event.type == pygame.KEYDOWN
                and event.key == pygame.K_SPACE
            ):
                movefunc()
            elif (
                not is_player
                and event.type == pygame.KEYDOWN
                and event.key == pygame.K_a
            ):
                self.autoPlay = not self.autoPlay

        now = pygame.time.get_ticks()
        if self.autoPlay and now - self.last >= 300:
            self.last = pygame.time.get_ticks()
            movefunc()

        self.want_to_quit = False

    def playHumanGame(self, fpsclock):
        """
        Play the game.

        :param fpsclock: Track time, Clock object.
        """
        while not self.want_to_quit:
            dt = fpsclock.tick()
            self.screen.fill((0, 0, 0))
            self.draw()
            self.drawShortcuts(True, True)
            pygame.display.flip()
            self.catchGameEvents(True, None)
            self.update(dt)
            self.checkGameState(False)

    def checkGameState(self, is_AI):
        """
        Check if the game is won. If it is won, we ask to the player if he want
        the play again, quit the game or want to go to the main menu.

        :param fpsclock:    Track time, Clock object.
        :param is_AI:       A boolean value to check if it is a game with a player or with an AI.
        :return:            Return False if the game is won or if the player want
                            to play again. Otherwise, False.
        """
        if self.isWin():
            self.want_to_quit = self.exitMenu(is_AI)
            return True
        return False

    def selectPlayerMenu(self, AI_type):
        """
        Ask to the player if he wants to play or if he wants an AI to play.

        :param AI_type  A string representing which type of AI you are using
        :return:        Return a String that reprensent the choice of the player.
                        It returns "human" or "AI".
        """
        self.screen.fill((0, 0, 0))
        self.drawText(AI_type, TXT_MENU_SIZE, 400, 50, 255, 255, 255, True)
        self.drawText(
            "Press <h> to play", TXT_MENU_SIZE, 400, 150, 255, 255, 255, True
        )
        self.drawText(
            "Press <a> to run the AI",
            TXT_MENU_SIZE,
            400,
            250,
            255,
            255,
            255,
            True,
        )
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        self.shuffle()
                        return "human"
                    if event.key == pygame.K_a:
                        return "AI"

    def selectModel(self):
        """
        Ask to the player if he wants to play with a new AI or a pre-existing AI.

        :return:        Return a String that reprensents the name of the AI selected.
                        It returns "" if user wants to stay in menu.
        """
        model_name = ""
        # Check if the AI folder exists, if not, we create this folder.
        if not os.path.exists("QTable"):
            os.makedirs("QTable")
        # Take all existing AI in the repo /QTable.
        onlyfiles = [
            f
            for f in os.listdir("QTable/")
            if os.path.isfile(os.path.join("QTable/", f))
        ]
        # Do a natural sort, it is a sort where we sort as follow:
        # [a1, a11, a2, a22] => [a1, a2, a11, a22].
        reg = re.compile(r"QTable_(\d+).txt")
        onlyfiles = list(filter(lambda f: reg.search(f), onlyfiles))
        print(onlyfiles)
        sorted_models = sorted([int(reg.search(f).group(1)) for f in onlyfiles])
        print(sorted_models)

        while True:
            self.screen.fill((0, 0, 0))
            self.drawText(
                "Do you want to create a new AI ?",
                TXT_MENU_SIZE,
                400,
                150,
                255,
                255,
                255,
                True,
            )
            self.drawText(
                "Press <y> for yes",
                TXT_MENU_SIZE,
                400,
                250,
                255,
                255,
                255,
                True,
            )
            self.drawText(
                "Press <n> for no", TXT_MENU_SIZE, 400, 290, 255, 255, 255, True
            )
            pygame.display.flip()
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.KEYDOWN:
                    # Create a new AI.
                    if event.key == pygame.K_y:
                        # model_name = "QTable/QTable"
                        model_name = pathlib.Path("QTable")
                        # Check if there exists an AI model.
                        if onlyfiles != []:
                            if len(onlyfiles) < 36:
                                return str(
                                    model_name
                                    / pathlib.Path(
                                        f"QTable_{sorted_models[-1]+1}.txt"
                                    )
                                )
                            else:
                                self.diplayFullModels()
                                return ""
                        else:
                            return str(
                                model_name / pathlib.Path("QTable_0.txt")
                            )
                    if event.key == pygame.K_n:
                        return self.selectExistingModel(onlyfiles)
                    if event.key == pygame.K_ESCAPE:
                        if self.pauseMenu() == True:
                            # We return the model_name because it is egal to "".
                            # It means that the player does not want to play.
                            return str(model_name)

    def diplayFullModels(self):
        """
        Diplay an error message if there are to many saved AIs.
        """
        self.screen.fill((0, 0, 0))
        self.drawText(
            "There are to many saved AI models in 'QTable' folder.",
            TXT_CORE_SIZE,
            400,
            150,
            255,
            255,
            255,
            True,
        )
        self.drawText(
            "Delete one of them to create a new AI agent.",
            TXT_CORE_SIZE,
            400,
            200,
            255,
            255,
            255,
            True,
        )
        self.drawText(
            "Press any key to return to menu.",
            TXT_CORE_SIZE,
            400,
            250,
            255,
            255,
            255,
            True,
        )
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    return
                self.catchExitEvent(event)

    def drawLabels(self, strList, starting_x, starting_y, dx=200, dy=30):
        """
        Return and draws a list of labels, labeled by the given list of strings.

        :param strList:     A list of strings with which the labels will be filled.
        :param starting_x:  Integer corresponding to the x position of the first label.
        :param starting_y:  Integer corresponding to the y position of the first label.
        :param dx:          Integer corresponding to the x spacing between labels.
        :param dy:          Integer corresponding to the y spacing between labels.
        :return:            Return a list of Label objects.
        """
        initial_y = starting_y
        labels = []
        # Create a clickable label for each existing file.
        for string in strList:
            labels.append(Label(string, starting_x, starting_y))
            self.screen.blit(labels[-1].getSurface(), (starting_x, starting_y))
            # Adjust the position for the next clickable label.
            if starting_y + dy > HEIGHT - dy:
                starting_x += dx
                starting_y = initial_y
            else:
                starting_y += dy
        return labels

    def drawExistingModels(self, onlyfiles):
        """
        Show all AI that the player has in the folder QTable. If the player has nothing,
        we ask him to go the menu.

        :param onlyfiles:   A list of String of all files in the repo QTable that contains all AI models.
        :return:            Return a String that represents the name of the AI selected.
                            It returns "" if user wants to go to menu.
        """
        while True:
            self.screen.fill((0, 0, 0))
            self.drawText(
                "Select an AI to load",
                TXT_MENU_SIZE,
                400,
                15,
                255,
                255,
                255,
                True,
            )
            labels = self.drawLabels(onlyfiles, 100, 65, 200, 35)
            pygame.display.flip()
            # Catch events.
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.mouse.get_pressed()[0]:
                        mx, my = pygame.mouse.get_pos()
                        for label in labels:
                            # Check if the location of the click is on the label.
                            if label.rect.collidepoint(mx, my):
                                return "QTable/" + label.getText()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.pauseMenu() == True:
                            # We return "" because it means that the player
                            # does not want to play.
                            return ""

    def drawNonExistingModel(self):
        """
        The player has no AI model on his computer. He needs to go to the main menu or he can
        exit the program.

        :return:            Return a String that represents the name of the AI selected.
                            It returns "" if user wants to go to menu.
        """
        self.screen.fill((0, 0, 0))
        self.drawText(
            "There is not saved AI.",
            TXT_MENU_SIZE,
            400,
            150,
            255,
            255,
            255,
            True,
        )
        self.drawText(
            "Press <m> to go to the menu.",
            TXT_MENU_SIZE,
            400,
            250,
            255,
            255,
            255,
            True,
        )
        pygame.display.flip()
        while True:
            # Catch events.
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        return ""

    def selectExistingModel(self, onlyfiles):
        """
        Show all AI that the player has in the folder QTable. If the player has nothing,
        we ask him to go the menu.

        :param onlyfiles:   A list of String of all files in the repo QTable that contains all AI models.
        :return:            Return a String that represents the name of the AI selected.
                            It returns "" if user wants to stay in menu.
        """
        # Check if there exists an AI model.
        if onlyfiles != []:
            return self.drawExistingModels(onlyfiles)
        # If not, the player need to go the menu or he can quit the application.
        else:
            return self.drawNonExistingModel()

    def selectTrainingNb(self):
        """
        Diplay the screen to select the number of games to train the AI on.

        :return:                Integer corresponding to the number of trainings.
        """
        trainingNb = ["1000", "10000", "50000", "100000", "200000"]
        while True:
            self.screen.fill((0, 0, 0))
            self.drawText(
                "Choose the number of games to train the AI on.",
                TXT_MENU_SIZE,
                400,
                15,
                255,
                255,
                255,
                True,
            )
            labels = self.drawLabels(trainingNb, 40, 150, 100, 40)
            pygame.display.flip()
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.mouse.get_pressed()[0]:
                        mx, my = pygame.mouse.get_pos()
                        for label in labels:
                            # Check if the location of the click is on the label.
                            if label.rect.collidepoint(mx, my):
                                return int(label.getText())
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.pauseMenu() == True:
                            # The player wants to return to the menu.
                            return 0

    def trainingDiplay(
        self,
        title,
        trainingGames,
        totalGames,
        trainingInterval,
        totalAvg,
        localAvg,
    ):
        """
        Diplay of the training stats.

        :param title:           String corresponding to the title of the screen.
        :param trainingGames:   Integer corresponding to the games played so far in the training.
        :param totalGames:      Integer corresponding to the total number of games played by the AI.
        :param trainingInterval:Integer corresponding to the number of games played after the last refresh.
        :param totalAvg:        Double corresponding to the average number of moves per game, for all the training games.
        :param totalAvg:        Double corresponding to the average number of moves per game,
                                for the games played after the last refresh.
        """
        self.screen.fill((0, 0, 0))
        self.drawText(title, TXT_MENU_SIZE, 400, 15, 255, 255, 255, True)
        self.drawText(
            "Games solved for this training: {}".format(trainingGames),
            TXT_CORE_SIZE,
            30,
            200,
            255,
            255,
            255,
            False,
        )
        self.drawText(
            "Total number of games: {}".format(totalGames),
            TXT_CORE_SIZE,
            30,
            250,
            255,
            255,
            255,
            False,
        )
        self.drawText(
            "Average number of moves for the training: {}".format(totalAvg),
            TXT_CORE_SIZE,
            30,
            300,
            255,
            255,
            255,
            False,
        )
        self.drawText(
            "Average number of moves for last {} games: {}".format(
                trainingInterval, localAvg
            ),
            TXT_CORE_SIZE,
            30,
            350,
            255,
            255,
            255,
            False,
        )
        pygame.display.flip()

    def pauseMenu(self):
        """
        Ask to the player if he want to continue the game or if he want to go to the main menu.

        :return:         Return True if the player want to go to the main menu.
        """
        self.screen.fill((0, 0, 0))
        self.drawText(
            "Do you want to go back",
            TXT_MENU_SIZE,
            400,
            100,
            255,
            255,
            255,
            True,
        )
        self.drawText(
            "to the main menu ?", TXT_MENU_SIZE, 400, 140, 255, 255, 255, True
        )
        self.drawText(
            "Press <y> for yes", TXT_MENU_SIZE, 400, 250, 255, 255, 255, True
        )
        self.drawText(
            "Press <n> for no (or escape)",
            TXT_MENU_SIZE,
            400,
            290,
            255,
            255,
            255,
            True,
        )
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        return True
                    if event.key == pygame.K_n or event.key == pygame.K_ESCAPE:
                        return False

    def drawExitMenuAI(self):
        """
        Draw text for the exit menu of a AI game.
        """
        self.drawText("The AI won!", TXT_MENU_SIZE, 250, 80, 0, 0, 0, True)
        self.drawText(
            "Congratulations !", TXT_MENU_SIZE, 250, 160, 0, 0, 0, True
        )
        self.drawText(
            "Moves : {}".format(self.nb_move),
            TXT_CORE_SIZE,
            500,
            10,
            255,
            255,
            255,
            False,
        )
        self.drawText("Shortcuts", TXT_CORE_SIZE, 500, 40, 255, 255, 255, False)
        self.drawText(
            "New random board : <y>",
            TXT_CORE_SIZE,
            500,
            70,
            255,
            255,
            255,
            False,
        )
        self.drawText(
            "Menu : <m>", TXT_CORE_SIZE, 500, 130, 255, 255, 255, False
        )
        self.drawText(
            "Quit : <n>", TXT_CORE_SIZE, 500, 160, 255, 255, 255, False
        )

    def drawExitMenuHuman(self):
        """
        Draw text for the exit menu of a player game.
        """
        self.drawText("You won!", TXT_MENU_SIZE, 250, 80, 0, 0, 0, True)
        self.drawText(
            "Congratulations !", TXT_MENU_SIZE, 250, 160, 0, 0, 0, True
        )
        self.drawText(
            "Moves : {}".format(self.nb_move),
            TXT_CORE_SIZE,
            500,
            10,
            255,
            255,
            255,
            False,
        )
        self.drawText("Shortcuts", TXT_CORE_SIZE, 500, 40, 255, 255, 255, False)
        self.drawText(
            "Restart : <y>", TXT_CORE_SIZE, 500, 70, 255, 255, 255, False
        )
        self.drawText(
            "Menu : <m>", TXT_CORE_SIZE, 500, 100, 255, 255, 255, False
        )
        self.drawText(
            "Quit : <n>", TXT_CORE_SIZE, 500, 130, 255, 255, 255, False
        )

    def exitMenu(self, is_AI):
        """
        The menu to exit, restart the game or go to the main menu.

        :param fpsclock:    Track time, Clock object.
        :param is_AI:       A boolean value to check if it is a game with a player or with an AI.
        :return:            Return True if the player want to go to the main menu.
        """
        self.screen.fill((0, 0, 0))
        self.rect = pygame.Rect(
            0,
            0,
            self.gs[0] * (self.ts + self.ms) + self.ms,
            self.gs[1] * (self.ts + self.ms) + self.ms,
        )
        self.pic = pygame.transform.smoothscale(
            pygame.image.load(
                pathlib.Path("assets") / pathlib.Path("bluredImage.png")
            ),
            self.rect.size,
        )
        self.screen.blit(self.pic, self.rect)
        if is_AI:
            self.drawExitMenuAI()
        else:
            self.drawExitMenuHuman()
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        self.shuffle()
                        self.nb_move = 0
                        return False
                    if event.key == pygame.K_n:
                        self.exit()
                    if event.key == pygame.K_m:
                        return True

    def playTrainMenu(self):
        """
        The menu to choose if you want to play with or train the AI.

        :return:            Boolean, False if we want to play, True if we want to train the AI,
                            None if we want to go to the main menu.
        """
        finished = False
        while not finished:
            self.screen.fill((0, 0, 0))
            self.drawText(
                "Play or resume training the AI ?",
                TXT_MENU_SIZE,
                400,
                150,
                255,
                255,
                255,
                True,
            )
            self.drawText(
                "Press <p> to play.",
                TXT_MENU_SIZE,
                400,
                250,
                255,
                255,
                255,
                True,
            )
            self.drawText(
                "Press <t> to train.",
                TXT_MENU_SIZE,
                400,
                290,
                255,
                255,
                255,
                True,
            )
            pygame.display.flip()
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        return False
                    if event.key == pygame.K_t:
                        return True
                    if event.key == pygame.K_ESCAPE:
                        self.want_to_quit = self.pauseMenu()
                        finished = self.want_to_quit

    def convertToString(self):
        """
        Converts the current board into a string that can be feeded to AI.

        :return: Return a String of the board.
        """
        state = []
        for i in range(9):
            state.append(0)
        for j in range(self.tiles_len + 1):
            state[self.tiles[j][0] + 3 * self.tiles[j][1]] = str(j + 1)
        return "".join(state)

    def exit(self):
        """
        Exit the application.
        """
        pygame.quit()
        sys.exit()


class Label:
    def __init__(self, text, x, y):
        """
        Init a label.

        :param text:    The text for the label, String.
        :param x:       The x position for the label, Int.
        :param y:       The y position for the label, Int.
        """
        self.x = x
        self.y = y
        self.font = pygame.font.Font(None, 40)
        self.originalText = text
        self.text = self.font.render(text, 1, pygame.Color("White"))
        self.size = self.w, self.h = self.text.get_size()
        self.rect = pygame.Rect(self.x, self.y, self.w, self.h)
        self.surface = pygame.Surface(self.size)
        self.surface.blit(self.text, (0, 0))
        self.label_clicked = False

    def getSurface(self):
        """
        Getter for the surface of the label.

        :return: Return the surface of the label, Surface object.
        """
        return self.surface

    def getText(self):
        """
        Getter for the original text of the label.

        :return: Return a String, it is the original text of the label.
        """
        return self.originalText

    def isClicked(self):
        """
        Check if the label has been clicked

        :return: Return a Boolean, it is True if the label has been clicked, otherwise False.
        """
        return self.label_clicked

    def clicked(self):
        """
        The label has been clicked.
        """
        self.label_clicked = True
