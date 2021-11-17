import enum

EMPTY = 0
ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4

MONTE_CARLO = "MONTE_CARLO"
MINIMAX = "MINIMAX"
RANDOM = "RANDOM"
RANDOM_IMPR = "RANDOM_IMPR"

SQUARE_SIZE = 100


class Event(enum.Enum):
    PIECE_PLACED = 1
    GAME_WON = 2
    GAME_RESET = 3


class Observable:
    def __init__(self):
        self._observers = []

    def notify(self, event, *argv):
        for obs in self._observers:
            obs.update(self, event, *argv)

    def add_observer(self, obs):
        self._observers.append(obs)

    def remove_observer(self, obs):
        if obs in self._observers:
            self._observers.remove(obs)


class Observer:
    def __init__(self):
        pass

    def update(self, obj, event, *argv):
        pass
