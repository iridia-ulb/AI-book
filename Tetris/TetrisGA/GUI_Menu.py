from dataclasses import dataclass
import pygame_gui
import pygame
from pygame_gui.elements.ui_text_box import UITextBox
from pygame_gui.elements.ui_text_entry_line import UITextEntryLine
from pygame_gui.elements.ui_selection_list import UISelectionList
from pygame_gui.windows.ui_file_dialog import UIFileDialog


@dataclass
class Menu:
    """
    Main class used to define a menu in our Tetris Training Framework
    """

    manager: pygame_gui.UIManager
    window_surface: pygame.Surface
    background: pygame.Surface
    screen_width: int
    screen_height: int
    dimensions: (int, int)
    background_color: pygame.Color
    caption: str

    def __init__(self, screen_width, screen_height, color_str: str, caption):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.background_color = pygame.Color(color_str)
        self.caption = caption
        pygame.display.set_caption(caption)
        self.dimensions = (self.screen_width, self.screen_height)
        self.manager = pygame_gui.UIManager((self.screen_width, self.screen_height))
        self.window_surface = pygame.display.set_mode(size=(self.screen_width, self.screen_height))
        self.background = pygame.Surface(self.dimensions)
        self.background.fill(self.background_color)

    def initialize_entry_line(self, text: str, x: int, y: int, width=200, height=50):

        text_box = UITextBox(
            html_text=text, relative_rect=pygame.Rect((x, y), (width, height)), manager=self.manager
        )
        entry_line = UITextEntryLine(
            relative_rect=pygame.Rect((x + 250, y), (width / 2, height)), manager=self.manager
        )
        return text_box, entry_line

    def initialize_selection(
        self, text: str, x: int, y: int, item_list: list, default: list, width=200, height=50
    ):

        text_box = UITextBox(
            html_text=text, relative_rect=pygame.Rect((x, y), (width, height)), manager=self.manager
        )
        selection_list = UISelectionList(
            relative_rect=pygame.Rect((x + 200, y), (width, 170)),
            item_list=item_list,
            manager=self.manager,
            allow_multi_select=True,
            default_selection=default,
        )
        return text_box, selection_list

    def initialize_button(self, text: str, x: int, y: int, width=100, height=50):
        return pygame_gui.elements.ui_button.UIButton(
            relative_rect=pygame.Rect((x, y), (width, height)), text=text, manager=self.manager
        )

    def handle_events(self, event, is_running):
        if event.type == pygame.QUIT:
            is_running = False
        return is_running

    def run(self):
        pygame.init()
        clock = pygame.time.Clock()
        is_running = True

        while is_running:
            time_delta = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                is_running = self.handle_events(event, is_running)
                self.manager.process_events(event)

            self.manager.update(time_delta)

            self.window_surface.blit(self.background, (0, 0))
            self.manager.draw_ui(self.window_surface)

            pygame.display.update()
