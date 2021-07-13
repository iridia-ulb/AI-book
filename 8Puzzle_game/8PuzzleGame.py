import sys, os, random
import re
import pathlib
import pygame
import Label
import AI

# Constant for the size of the screen.
WIDTH = 800
HEIGHT = 500
# Constant for the size of the text.
TXT_CORE_SIZE = 38
TXT_MENU_SIZE = 50


class SlidePuzzle:
    def __init__(self, gs, ts, ms):
        """
        Init the game.

        :param gs: The grid size. It is a tuple (n,n) of Int.
        :param ts: The size of the tiles. It is an Int.
        :param ms: The size of the margin. It is an Int.
        """
        # Attributes for the main core of the game.

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
        self.rect = pygame.Rect(0, 0, gs[0] * (ts + ms) + ms, gs[1] * (ts + ms) + ms)
        self.pic = pygame.transform.smoothscale(
            pygame.image.load(pathlib.Path("assets") / pathlib.Path("image.png")), self.rect.size
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

        # Attributes for the AI.
        self.playerAI = None
        self.qValues = None

    def initPlayerAI(self, screen, modelAI, nbGames=100):
        """
        Initialise the AI player.

        :param screen:      The screen, Surface object.
        :param modelAI:     A string corresponding to the path of the QTable to load.
        :param nbGames:     Integer corresponding to the number of games (per type of game),
                            we want to train the AI on.
        """
        self.playerAI = AI.AIPlayer(modelAI)
        if not os.path.isfile(modelAI):
            pos = (250, 250)
            size = (300, 40)
            borderC = (255, 255, 255)
            barC = (255, 255, 255)

            self.drawBar(screen, pos, size, borderC, barC, 0.0)

            # Pre-Learning.
            for i in range(31):
                for j in range(nbGames):
                    self.playerAI.initLearning(i + 1, 1)
                    self.drawBar(
                        screen,
                        pos,
                        size,
                        borderC,
                        barC,
                        (i * nbGames + j + 1) / (31 * nbGames),
                    )
                    for event in pygame.event.get():
                        self.catchExitEvent(event)
            self.playerAI.saveQTable()

    def drawBar(self, screen, pos, size, borderC, barC, progress):
        """
        Draw the loading bar screen.

        :param screen:      The screen, Surface object.
        :param pos:         2-Tuple of integers corresponding to the (x,y) position of the bar.
        :param size:        2-Tuple of integers corresponding to the size of the bar.
        :param borderC:     3-Tuple of integers corresponding to the RGB color of bar's border.
        :param barC:        3-Tuple of integers corresponding to the RGB color of the bar.
        :param progress:    A double corresponding to the percentage of the progression.
        """
        screen.fill((0, 0, 0))
        self.drawText(
            screen,
            "The AI is pre-training, please wait ...",
            TXT_MENU_SIZE,
            400,
            130,
            255,
            255,
            255,
            True,
        )
        pygame.draw.rect(screen, borderC, (*pos, *size), 1)
        innerPos = (pos[0] + 3, pos[1] + 3)
        innerSize = ((size[0] - 6) * progress, size[1] - 6)
        pygame.draw.rect(screen, barC, (*innerPos, *innerSize))
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

            self.tilepos[i] = (X if abs(dx) < s else x + s if dx > 0 else x - s), (
                Y if abs(dy) < s else y + s if dy > 0 else y - s
            )

    def draw(self, screen):
        """
        Draw the game with the number of move.

        :param screen: The current screen, Surface object.
        """
        for i in range(self.tiles_len):
            x, y = self.tilepos[i]
            screen.blit(self.images[i], (x, y))
        self.drawText(
            screen,
            "Moves : " + str(self.nb_move),
            TXT_CORE_SIZE,
            500,
            10,
            255,
            255,
            255,
            False,
        )

    def drawText(self, screen, text, size, x, y, R, G, B, center):
        """
        Draw text.

        :param screen:  The screen, Surface object.
        :param text:    The text to draw on the screen, String.
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
            screen.blit(text, text_rect)
        else:
            screen.blit(text, (x, y))

    def drawShortcuts(self, screen, is_player):
        """
        Draw in game shortcuts.

        :param screen:      The screen, Surface object.
        :param is_player:   A Boolean, it checks if it is a player because, shorcuts are different in
                            the player mode or in the AI mode.
        """
        self.drawText(screen, "Shortcuts", TXT_CORE_SIZE, 500, 40, 255, 255, 255, False)
        self.drawText(
            screen, "Pause : Escape", TXT_CORE_SIZE, 500, 70, 255, 255, 255, False
        )
        if is_player:
            self.drawText(
                screen, "Move up : <z>", TXT_CORE_SIZE, 500, 100, 255, 255, 255, False
            )
            self.drawText(
                screen, "Move down : <s>", TXT_CORE_SIZE, 500, 130, 255, 255, 255, False
            )
            self.drawText(
                screen, "Move left : <q>", TXT_CORE_SIZE, 500, 160, 255, 255, 255, False
            )
            self.drawText(
                screen,
                "Move right : <d>",
                TXT_CORE_SIZE,
                500,
                190,
                255,
                255,
                255,
                False,
            )
            self.drawText(
                screen,
                "Random move : <Space>",
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
                screen, "AI move : <Space>", TXT_CORE_SIZE, 500, 100, 255, 255, 255, False
            )
            self.drawText(
                screen, "auto AI move : <a>", TXT_CORE_SIZE, 500, 130, 255, 255, 255, False
            )
            self.drawText(
                screen,
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
                screen,
                "UP : " + "{0:.2E}".format(self.qValues[0]),
                TXT_CORE_SIZE,
                500,
                220,
                255,
                255,
                255,
                False,
            )
            self.drawText(
                screen,
                "RIGHT : " + "{0:.2E}".format(self.qValues[1]),
                TXT_CORE_SIZE,
                500,
                250,
                255,
                255,
                255,
                False,
            )
            self.drawText(
                screen,
                "DOWN : " + "{0:.2E}".format(self.qValues[2]),
                TXT_CORE_SIZE,
                500,
                280,
                255,
                255,
                255,
                False,
            )
            self.drawText(
                screen,
                "LEFT : " + "{0:.2E}".format(self.qValues[3]),
                TXT_CORE_SIZE,
                500,
                310,
                255,
                255,
                255,
                False,
            )

    def playEvents(self, event, label=None):
        """
        Catch events from the mouse and the keyboard.
        Binded keys:
            - z moves the tile upwards
            - s moves the tile downwards
            - q moves the tile to the left
            - d moves the tile to the right
            - space choose a random mouvement

        :param event:   The current event, Event object.
        :param label:   Optional parameter, it is used for the creation of a board.
                        Label object.
        """
        mouse = pygame.mouse.get_pressed()
        mpos = pygame.mouse.get_pos()
        # If we use the left click
        if mouse[0]:
            # If we are in the creation mode, we check if the label validate has been clicked
            if label != None:
                if label.rect.collidepoint(mpos[0], mpos[1]):
                    label.clicked()
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
                for i in range(1000):
                    self.random()

    def catchExitEvent(self, event):
        """
        Check if it is a quit event.

        :param event:   Event object.
        """
        if event.type == pygame.QUIT:
            self.exit()

    def catchGameEvents(self, is_player, is_making_board, screen, label=None):
        """
        Catchs event during the game and during the board creation.

        :param is_player:       A boolean value to check if it is a game with a player or with an AI.
        :param is_making_board: A boolean value to check if the player is creating a board.
        :param screen:          The screen, Surface object.
        :param label:           Optional parameter, it is used for the creation of a board.
                                Label object.
        :return:                Return True if the player want to quit the game.
                                Otherwise, False.
        """
        for event in pygame.event.get():
            self.catchExitEvent(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.want_to_quit = self.pauseMenu(screen)
                    return
            if is_making_board:
                self.playEvents(event, label)
            elif is_player:
                self.playEvents(event)
            else:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.moveTileAI()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    self.autoPlay = not self.autoPlay
        now = pygame.time.get_ticks()
        if self.autoPlay and now - self.last >= 300:
            self.last = pygame.time.get_ticks()
            self.moveTileAI()

        self.want_to_quit = False

    def moveTileAI(self):
        """
        Update the board according to the next move of AI.
        """
        move = self.playerAI.getNextMove()
        x, y = self.opentile
        if move == self.playerAI.UP:
            tile = x, y - 1
        elif move == self.playerAI.RIGHT:
            tile = x + 1, y
        elif move == self.playerAI.DOWN:
            tile = x, y + 1
        else:
            tile = x - 1, y
        self.switch(tile, True)
        self.qValues = self.playerAI.getQValues(self.convertToString())

    def playAIGame(self, fpsclock, screen):
        """
        Play the game with AI.

        :param fpsclock:    Track time, Clock object.
        :param screen:      The screen, Surface object.
        """
        self.playerAI.playGame(self.convertToString())
        self.qValues = self.playerAI.getQValues(self.convertToString())
        while not self.want_to_quit:
            dt = fpsclock.tick()
            screen.fill((0, 0, 0))
            self.draw(screen)
            self.drawShortcuts(screen, False)
            pygame.display.flip()
            self.catchGameEvents(False, False, screen)
            self.update(dt)
            if self.checkGameState(fpsclock, screen, True):
                self.playerAI.playGame(self.convertToString())
                self.qValues = self.playerAI.getQValues(self.convertToString())

    def playHumanGame(self, fpsclock, screen):
        """
        Play the game.

        :param fpsclock: Track time, Clock object.
        :param screen:   The screen, Surface object.
        """
        while not self.want_to_quit:
            dt = fpsclock.tick()
            screen.fill((0, 0, 0))
            self.draw(screen)
            self.drawShortcuts(screen, True)
            pygame.display.flip()
            self.catchGameEvents(True, False, screen)
            self.update(dt)
            self.checkGameState(fpsclock, screen, False)

    def checkGameState(self, fpsclock, screen, is_AI):
        """
        Check if the game is won. If it is won, we ask to the player if he want
        the play again, quit the game or want to go to the main menu.

        :param fpsclock:    Track time, Clock object.
        :param screen:      The screen, Surface object.
        :param is_AI:       A boolean value to check if it is a game with a player or with an AI.
        :return:            Return False if the game is won or if the player want
                            to play again. Otherwise, False.
        """
        if self.isWin():
            self.want_to_quit = self.exitMenu(fpsclock, screen, is_AI)
            return True
        return False

    def selectPlayerMenu(self, screen):
        """
        Ask to the player if he wants to play or if he wants an AI to play.

        :param screen:  The screen, Surface object.
        :return:        Return a String that reprensent the choice of the player.
                        It returns "human" or "AI".
        """
        screen.fill((0, 0, 0))
        self.drawText(
            screen, "Press <h> to play", TXT_MENU_SIZE, 400, 150, 255, 255, 255, True
        )
        self.drawText(
            screen,
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

    def selectModel(self, screen):
        """
        Ask to the player if he wants to play with a new AI or a pre-existing AI.

        :param screen:  The screen, Surface object.
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
        sorted_models = sorted( [ int(reg.search(f).group(1)) for f in onlyfiles ])
        print(sorted_models)

        while True:
            screen.fill((0, 0, 0))
            self.drawText(
                screen,
                "Do you want to play with a new AI or not ?",
                TXT_MENU_SIZE,
                400,
                150,
                255,
                255,
                255,
                True,
            )
            self.drawText(
                screen,
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
                screen, "Press <n> for no", TXT_MENU_SIZE, 400, 290, 255, 255, 255, True
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
                                return str(model_name / pathlib.Path(f"QTable_{sorted_models[-1]+1}.txt"))
                            else:
                                self.diplayFullModels(screen)
                                return ""
                        else:
                            return str(model_name / pathlib.Path("QTable_0.txt"))
                    if event.key == pygame.K_n:
                        return self.selectExistingModel(screen, onlyfiles)
                    if event.key == pygame.K_ESCAPE:
                        if self.pauseMenu(screen) == True:
                            # We return the model_name because it is egal to "".
                            # It means that the player does not want to play.
                            return str(model_name)

    def diplayFullModels(self, screen):
        """
        Diplay an error message if there are to many saved AIs.

        :param screen:  The screen, Surface object.
        """
        screen.fill((0, 0, 0))
        self.drawText(
            screen,
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
            screen,
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
            screen,
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

    def drawLabels(self, screen, strList, starting_x, starting_y, dx=200, dy=30):
        """
        Return and draws a list of labels, labeled by the given list of strings.

        :param screen:      The screen, Surface object.
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
            labels.append(Label.Label(string, starting_x, starting_y))
            screen.blit(labels[-1].getSurface(), (starting_x, starting_y))
            # Adjust the position for the next clickable label.
            if starting_y + dy > HEIGHT - dy:
                starting_x += dx
                starting_y = initial_y
            else:
                starting_y += dy
        return labels

    def drawExistingModels(self, screen, onlyfiles):
        """
        Show all AI that the player has in the folder QTable. If the player has nothing,
        we ask him to go the menu.

        :param screen:      The screen, Surface object.
        :param onlyfiles:   A list of String of all files in the repo QTable that contains all AI models.
        :return:            Return a String that represents the name of the AI selected.
                            It returns "" if user wants to go to menu.
        """
        while True:
            screen.fill((0, 0, 0))
            self.drawText(
                screen,
                "Click on the AI you want",
                TXT_MENU_SIZE,
                400,
                15,
                255,
                255,
                255,
                True,
            )
            labels = self.drawLabels(screen, onlyfiles, 100, 65, 200, 35)
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
                        if self.pauseMenu(screen) == True:
                            # We return "" because it means that the player
                            # does not want to play.
                            return ""

    def drawNonExistingModel(self, screen):
        """
        The player has no AI model on his computer. He needs to go to the main menu or he can
        exit the program.

        :param screen:      The screen, Surface object.
        :return:            Return a String that represents the name of the AI selected.
                            It returns "" if user wants to go to menu.
        """
        screen.fill((0, 0, 0))
        self.drawText(
            screen,
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
            screen,
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

    def selectExistingModel(self, screen, onlyfiles):
        """
        Show all AI that the player has in the folder QTable. If the player has nothing,
        we ask him to go the menu.

        :param screen:      The screen, Surface object.
        :param onlyfiles:   A list of String of all files in the repo QTable that contains all AI models.
        :return:            Return a String that represents the name of the AI selected.
                            It returns "" if user wants to stay in menu.
        """
        # Check if there exists an AI model.
        if onlyfiles != []:
            return self.drawExistingModels(screen, onlyfiles)
        # If not, the player need to go the menu or he can quit the application.
        else:
            return self.drawNonExistingModel(screen)

    def selectTrainingNb(self, screen):
        """
        Diplay the screen to select the number of games to train the AI on.

        :param screen:          The screen, Surface object.
        :return:                Integer corresponding to the number of trainings.
        """
        trainingNb = ["1000", "10000", "50000", "100000", "200000"]
        while True:
            screen.fill((0, 0, 0))
            self.drawText(
                screen,
                "Choose the number of games to train the AI.",
                TXT_MENU_SIZE,
                400,
                15,
                255,
                255,
                255,
                True,
            )
            labels = self.drawLabels(screen, trainingNb, 40, 150, 100, 40)
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
                        if self.pauseMenu(screen) == True:
                            # The player wants to return to the menu.
                            return 0

    def trainingDiplay(
        self,
        screen,
        title,
        trainingGames,
        totalGames,
        trainingInterval,
        totalAvg,
        localAvg,
    ):
        """
        Diplay of the training stats.

        :param screen:          The screen, Surface object.
        :param title:           String corresponding to the title of the screen.
        :param trainingGames:   Integer corresponding to the games played so far in the training.
        :param totalGames:      Integer corresponding to the total number of games played by the AI.
        :param trainingInterval:Integer corresponding to the number of games played after the last refresh.
        :param totalAvg:        Double corresponding to the average number of moves per game, for all the training games.
        :param totalAvg:        Double corresponding to the average number of moves per game,
                                for the games played after the last refresh.
        """
        screen.fill((0, 0, 0))
        self.drawText(screen, title, TXT_MENU_SIZE, 400, 15, 255, 255, 255, True)
        self.drawText(
            screen,
            "Games solved for this training : " + str(trainingGames),
            TXT_CORE_SIZE,
            30,
            200,
            255,
            255,
            255,
            False,
        )
        self.drawText(
            screen,
            "Total number of games : " + str(totalGames),
            TXT_CORE_SIZE,
            30,
            250,
            255,
            255,
            255,
            False,
        )
        self.drawText(
            screen,
            "Average number of moves for the training : " + str(totalAvg),
            TXT_CORE_SIZE,
            30,
            300,
            255,
            255,
            255,
            False,
        )
        self.drawText(
            screen,
            "Average number of moves for last "
            + str(trainingInterval)
            + " games : "
            + str(localAvg),
            TXT_CORE_SIZE,
            30,
            350,
            255,
            255,
            255,
            False,
        )
        pygame.display.flip()

    def trainingAI(self, screen, trainingNb):
        """
        Launch and desplay the training of the AI.

        :param screen:      The screen, Surface object.
        :param trainingNb:  An integer giving the number of games to train the AI on.
        """
        playedGames = self.playerAI.getNbGames()
        self.trainingDiplay(screen, "The AI is training ...", 0, playedGames, 1, 0, 0)
        trainedSoFar = 0
        totalNbMoves = 0
        lastGamesNbMoves = 0
        while trainedSoFar < trainingNb:
            trainingInterval = self.getTrainingInterval(playedGames)
            nbMoves = self.playerAI.train(trainingInterval)
            totalNbMoves += nbMoves
            trainedSoFar += trainingInterval
            playedGames += trainingInterval
            self.trainingDiplay(
                screen,
                "The AI is training ...",
                trainedSoFar,
                playedGames,
                trainingInterval,
                round(totalNbMoves / trainedSoFar, 2),
                round(nbMoves / trainingInterval, 2),
            )
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.playerAI.saveQTable()
                    self.exit()
        self.playerAI.saveQTable()
        while True:
            self.trainingDiplay(
                screen,
                "Training finished, press <m> to access the menu.",
                trainedSoFar,
                playedGames,
                trainingInterval,
                round(totalNbMoves / trainedSoFar, 2),
                round(nbMoves / trainingInterval, 2),
            )
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    return

    def getTrainingInterval(self, x):
        """
        Compute the number of games to train on, before refreshing the screen,
        depending on the total number of games that was played by the AI.

        :param x:   Integer corresponding to total number of games that was played by the AI.
        :return:    Integer corresponding to number of games to train on.
        """
        if x < 1000:
            res = 1
        elif 1000 <= x < 100000:
            res = x / 1000
        elif 100000 <= x < 1000000:
            res = (2 * x / 9000) + (700 / 9)
        else:
            res = 300
        return int(res)

    def generateOwnBoard(self, fpsclock, screen):
        """
        Let the player generates a board for the AI.

        :param fpsclock:  Track time, Clock object.
        :param screen:    The screen, Surface object.
        """
        finished = False
        while not finished and not self.want_to_quit:
            dt = fpsclock.tick()
            screen.fill((0, 0, 0))
            self.draw(screen)
            self.drawShortcuts(screen, True)
            # Create a label for the validation of the board.
            validate_label = Label.Label("Validate", 600, 400)
            screen.blit(validate_label.getSurface(), (600, 400))
            pygame.display.flip()
            self.catchGameEvents(False, True, screen, validate_label)
            self.update(dt)
            # Check if the label is clicked.
            if validate_label.isClicked():
                self.nb_move = 0
                finished = True

    def selectBoard(self, fpsclock, screen):
        """
        The player can select a personnal board or he can generate a random board.

        :param fpsclock:  Track time, Clock object.
        :param screen:    The screen, Surface object.
        """
        finished = False
        while not finished:
            screen.fill((0, 0, 0))
            self.drawText(
                screen,
                "Do you want to generate your own board or not?",
                TXT_MENU_SIZE,
                400,
                150,
                255,
                255,
                255,
                True,
            )
            self.drawText(
                screen,
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
                screen, "Press <n> for no", TXT_MENU_SIZE, 400, 290, 255, 255, 255, True
            )
            pygame.display.flip()
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        self.generateOwnBoard(fpsclock, screen)
                        finished = True
                    if event.key == pygame.K_n:
                        self.shuffle()
                        finished = True
                    if event.key == pygame.K_ESCAPE:
                        finished = self.pauseMenu(screen)
                        self.want_to_quit = finished

    def pauseMenu(self, screen):
        """
        Ask to the player if he want to continue the game or if he want to go to the main menu.

        :param screen:   The screen, Surface object.
        :return:         Return True if the player want to go to the main menu.
        """
        screen.fill((0, 0, 0))
        self.drawText(
            screen,
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
            screen, "to the main menu ?", TXT_MENU_SIZE, 400, 140, 255, 255, 255, True
        )
        self.drawText(
            screen, "Press <y> for yes", TXT_MENU_SIZE, 400, 250, 255, 255, 255, True
        )
        self.drawText(
            screen,
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

    def drawExitMenuAI(self, screen):
        """
        Draw text for the exit menu of a AI game.

        :param screen:  The screen, Surface object.
        """
        self.drawText(screen, "The AI won!", TXT_MENU_SIZE, 250, 80, 0, 0, 0, True)
        self.drawText(
            screen, "Congratulations !", TXT_MENU_SIZE, 250, 160, 0, 0, 0, True
        )
        self.drawText(
            screen,
            "Moves : " + str(self.nb_move),
            TXT_CORE_SIZE,
            500,
            10,
            255,
            255,
            255,
            False,
        )
        self.drawText(screen, "Shortcuts", TXT_CORE_SIZE, 500, 40, 255, 255, 255, False)
        self.drawText(
            screen,
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
            screen, "New own board : <p>", TXT_CORE_SIZE, 500, 100, 255, 255, 255, False
        )
        self.drawText(
            screen, "Menu : <m>", TXT_CORE_SIZE, 500, 130, 255, 255, 255, False
        )
        self.drawText(
            screen, "Quit : <n>", TXT_CORE_SIZE, 500, 160, 255, 255, 255, False
        )

    def drawExitMenuHuman(self, screen):
        """
        Draw text for the exit menu of a player game.

        :param screen:  The screen, Surface object.
        """
        self.drawText(screen, "You won!", TXT_MENU_SIZE, 250, 80, 0, 0, 0, True)
        self.drawText(
            screen, "Congratulations !", TXT_MENU_SIZE, 250, 160, 0, 0, 0, True
        )
        self.drawText(
            screen,
            "Moves : " + str(self.nb_move),
            TXT_CORE_SIZE,
            500,
            10,
            255,
            255,
            255,
            False,
        )
        self.drawText(screen, "Shortcuts", TXT_CORE_SIZE, 500, 40, 255, 255, 255, False)
        self.drawText(
            screen, "Restart : <y>", TXT_CORE_SIZE, 500, 70, 255, 255, 255, False
        )
        self.drawText(
            screen, "Menu : <m>", TXT_CORE_SIZE, 500, 100, 255, 255, 255, False
        )
        self.drawText(
            screen, "Quit : <n>", TXT_CORE_SIZE, 500, 130, 255, 255, 255, False
        )

    def exitMenu(self, fpsclock, screen, is_AI):
        """
        The menu to exit, restart the game or go to the main menu.

        :param fpsclock:    Track time, Clock object.
        :param screen:      The screen, Surface object.
        :param is_AI:       A boolean value to check if it is a game with a player or with an AI.
        :return:            Return True if the player want to go to the main menu.
        """
        screen.fill((0, 0, 0))
        self.rect = pygame.Rect(
            0,
            0,
            self.gs[0] * (self.ts + self.ms) + self.ms,
            self.gs[1] * (self.ts + self.ms) + self.ms,
        )
        self.pic = pygame.transform.smoothscale(
            pygame.image.load(pathlib.Path("assets") / pathlib.Path("bluredImage.png")), self.rect.size
        )
        screen.blit(self.pic, self.rect)
        if is_AI:
            self.drawExitMenuAI(screen)
        else:
            self.drawExitMenuHuman(screen)
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                self.catchExitEvent(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        self.shuffle()
                        self.nb_move = 0
                        return False
                    if is_AI:
                        if event.key == pygame.K_p:
                            self.generateOwnBoard(fpsclock, screen)
                            return False
                    if event.key == pygame.K_n:
                        self.exit()
                    if event.key == pygame.K_m:
                        return True

    def playTrainMenu(self, screen):
        """
        The menu to choose if you want to play with or train the AI.

        :param screen:      The screen, Surface object.
        :return:            Boolean, False if we want to play, True if we want to train the AI,
                            None if we want to go to the main menu.
        """
        finished = False
        while not finished:
            screen.fill((0, 0, 0))
            self.drawText(
                screen,
                "Do you want to play or train the AI ?",
                TXT_MENU_SIZE,
                400,
                150,
                255,
                255,
                255,
                True,
            )
            self.drawText(
                screen,
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
                screen,
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
                        self.want_to_quit = self.pauseMenu(screen)
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


def main():
    """
    The main function to run the game.
    """
    pygame.init()
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.display.set_caption("8-Puzzle game")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    fpsclock = pygame.time.Clock()
    while True:
        game = SlidePuzzle((3, 3), 160, 5)
        choice = game.selectPlayerMenu(screen)
        if choice == "AI":
            modelAI = game.selectModel(screen)
            if modelAI != "":
                game.initPlayerAI(screen, modelAI)
                trainAI = game.playTrainMenu(screen)
                if trainAI:
                    trainingNb = game.selectTrainingNb(screen)
                    if trainingNb != 0:
                        game.trainingAI(screen, trainingNb)
                elif trainAI == False:
                    game.selectBoard(fpsclock, screen)
                    game.playAIGame(fpsclock, screen)
        else:
            game.playHumanGame(fpsclock, screen)
        del game


if __name__ == "__main__":
    main()
