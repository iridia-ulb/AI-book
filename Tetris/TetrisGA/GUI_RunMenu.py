from dataclasses import dataclass
import pygame_gui
import pygame
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from os.path import isfile

from GUI_Menu import Menu
from TetrisParallel import TetrisParallel

HEURISTIC_LABELS = [
    "Holes",
    "Height",
    "Bumpiness",
    "Line cleared",
    "Hollow columns",
    "Row Transition",
    "Column Transition",
    "Pitcount",
]
MAIN_HEURISTICS = HEURISTIC_LABELS[0:4]


@dataclass
class StartMenu(Menu):
    """
    Handle the creation of a Start Training Menu which extends the abstracts class Menu
    This menu is used to initialize the training parameters.
    """

    def __init__(self, max_training_time: int, screen_width, screen_height, color_str: str):
        super().__init__(screen_width, screen_height, color_str, "Start Menu")
        self.max_training_time = max_training_time
        self.nb_gen_entry = None
        self.time_limit_entry = None
        self.run_button = None
        self.heuristics_selection = None
        self.random_run = None
        self.error_text = pygame_gui.elements.ui_text_box.UITextBox(
            html_text="",
            relative_rect=pygame.Rect((200, 450), (300, 100)),
            manager=self.manager,
            visible=False,
        )
        self.init_commands()

    def init_commands(self):
        """
        Initialize the commands of the menu (button, entry) that defines the training parameters
        """
        heuristics_tb, self.heuristics_selection = self.initialize_selection(
            "Heuristics to consider", 100, 50, HEURISTIC_LABELS, MAIN_HEURISTICS
        )

        nb_gen_tb, self.nb_gen_entry = self.initialize_entry_line("Nb Generations", 100, 275)
        time_limit_tb, self.time_limit_entry = self.initialize_entry_line("Time limit", 100, 350)
        self.run_button = self.initialize_button("Run", 350, 600)

        self.path = str(Path("temp_train/"))

        self.nb_gen_entry.set_allowed_characters("numbers")
        self.time_limit_entry.set_allowed_characters("numbers")

    def handle_events(self, event, is_running):
        """
        Handle events linked to pressed button
        :param event: the event triggered
        :param is_running: a boolean value telling if the game is running
        :return: nothing
        """
        is_running = super(StartMenu, self).handle_events(event, is_running)
        if not is_running:
            return is_running

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.run_button:
                self.error_text.visible = False
                self.handle_run()

        return is_running

    def handle_run(self):
        """
        Handle the run event : launch the training
        """
        self.validate_nb_gen_entry()
        self.validate_time_entry()
        self.validate_heuristics()
        name_heuristics = self.heuristics_selection.get_multi_selection().copy()
        heuristics_to_consider = self.turn_heuristic_strings_into_indexes()
        print(heuristics_to_consider)
        if not self.error_text.visible:
            tetris_parallel = TetrisParallel(
                nb_gen=int(self.nb_gen_entry.text),
                limit_time=int(self.time_limit_entry.text),
                heuristics_selected=heuristics_to_consider,
                path=self.path,
                random_run=False,
                max_training_time=self.max_training_time,
            )
            tetris_parallel.launch()

            plot_training(self.path, int(self.nb_gen_entry.text), name_heuristics)
            pygame.display.set_mode(size=(self.screen_width, self.screen_height))

    def validate_path(self):
        """
        Check if the save directory for the training is given
        """
        if not self.path:
            self.display_error("please fill the path")

    def turn_heuristic_strings_into_indexes(self):
        """
        Converts heuristics to their indexes
        """
        weight_to_consider = []
        for heuristic in self.heuristics_selection.get_multi_selection():
            if heuristic == "Holes":
                weight_to_consider.append(0)
            elif heuristic == "Height":
                weight_to_consider.append(1)
            elif heuristic == "Bumpiness":
                weight_to_consider.append(2)
            elif heuristic == "Line cleared":
                weight_to_consider.append(3)
            elif heuristic == "Hollow columns":
                weight_to_consider.append(4)
            elif heuristic == "Row Transition":
                weight_to_consider.append(5)
            elif heuristic == "Column Transition":
                weight_to_consider.append(6)
            elif heuristic == "Pitcount":
                weight_to_consider.append(7)
        return weight_to_consider

    def validate_heuristics(self):
        """
        Throw an error if the used choosed no heuristics
        """
        if not self.heuristics_selection.get_multi_selection():
            self.display_error("Please, choose at least one heuristic to train your Genetic Agents")

    def display_error(self, error_text):
        """
        Display the error text given in parameters
        :param error_text: the error text to display
        """
        self.error_text.set_text(error_text)
        self.error_text.visible = True

    def validate_nb_gen_entry(self):
        """
        Display an error if the number of generation entered is not valid
        """
        if self.nb_gen_entry.text == "":
            self.display_error("Please write something for the number of gen")

        elif int(self.nb_gen_entry.text) > 1000:

            self.display_error("The number of generation cannot exceed 1000")

    def validate_time_entry(self):
        """
        Display an error if the time entry is not valid
        """
        if self.time_limit_entry.text == "":
            self.display_error("Please write something for the time limit")

        elif int(self.time_limit_entry.text) < 250 or int(self.time_limit_entry.text) > 5000:
            self.display_error("You cannot train your GA with a time lower 250 or above 5000")


def plot_training(path, nb_gen, heuristics):

    """
    Plot the evolution of the highest score of the
    :param path: the path of where to save the graphic file
    :param nb_gen: the number of generation
    :param heuristics: the heuristics considered
    """
    highest_score_run = []
    for i in range(1, nb_gen + 1):
        file_name = path + f"/model_gen_{i}.csv"
        if isfile(file_name):
            df = pd.read_csv(file_name)
            highest_score_run.append(df["score"].max())

    if len(highest_score_run) != 0:
        plt.xlabel("Generation")
        plt.xticks(range(1, nb_gen + 1))
        plt.ylabel("Highest score")
        plt.title(f"Using {heuristics}")
        plt.suptitle("Evolution highest score over generations")
        plt.plot(range(1, len(highest_score_run) + 1, 1), highest_score_run)
        plt.savefig(path + "/graph.jpg")
        plt.show()
