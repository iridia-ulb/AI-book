from dna import Dna
from gameModule import (
    RIGHT,
    LEFT,
    DOWN,
    UP,
    SNAKE_CHAR,
    FOOD_CHAR,
    WALL_CHAR,
)


class Snake:
    """
    Represents the AI that plays the Snake game.
    """

    def __init__(self, dna, hunger=100):
        """
        Constructor.
        dna (Dna): The DNA of the snake
        hunger (int): The starting hunger of the snake that decreases at each movement
        """
        self.dna = dna
        self.fitness = None
        self.hunger = hunger
        self.maxHunger = hunger
        self.nbrMove = 0
        self.previous_moves = []

    def mate(self, other, mutationRate=0.01):
        """
        Mate with another state to create a new snake.
        other (Snake): The other snake that mates with this one
        mutationRate (float): The probability for the DNA to mutate
        Returns the newly created snake (Snake)
        """
        newDna = self.dna.mix(other.dna, mutationRate)
        return Snake(newDna)

    def choose_next_move(self, state):
        """
        Choose a new move based on its vision.
        If the hunger of the snake is nul then return a string
        to indicate that the snakes is dead by starvation.
        vision (list): List containing distances between the head
        of the snake andother elements in the game
        Return the movement choice of the snake (tuple)
        """
        vision = self.get_simplified_state(state)
        if self.hunger > 0:
            self.hunger -= 1
            self.nbrMove += 1
            movesValues = self.dna.predict(vision)
            choice = 0

            movesValues = movesValues.tolist()

            # Chooses the best move (the move with the highest value)
            for i in range(1, len(movesValues[0])):
                if movesValues[0][i] > movesValues[0][choice]:
                    choice = i

            MOVEMENT = (RIGHT, LEFT, UP, DOWN)
            self.previous_moves.append(MOVEMENT[choice])
            if len(self.previous_moves) >= 3:
                self.previous_moves.pop(0)
            return MOVEMENT[choice]
        return "starve"

    def get_simplified_state(self, state):
        """
        returns a matrix of elements surrounding the snake and the preivous two
        moves, this serves as the input for the neural network.
        """
        res = self.get_line_elem(RIGHT, state)
        res += self.get_line_elem((DOWN[0], RIGHT[1]), state)
        res += self.get_line_elem(DOWN, state)
        res += self.get_line_elem((DOWN[0], LEFT[1]), state)
        res += self.get_line_elem(LEFT, state)
        res += self.get_line_elem((UP[0], LEFT[1]), state)
        res += self.get_line_elem(UP, state)
        res += self.get_line_elem((UP[0], RIGHT[1]), state)

        if len(self.previous_moves) == 0:
            res += [0, 0]
        elif len(self.previous_moves) == 1:  # previous previous move
            res += [
                self.previous_moves[0][0] / 2,
                self.previous_moves[0][1] / 2,
            ]
        else:
            res += [
                self.previous_moves[0][0] + self.previous_moves[1][0] / 2,
                self.previous_moves[0][1] + self.previous_moves[1][1] / 2,
            ]

        return res

    def is_in_grid(self, pos, grid):
        """
        Checks if an element is in the grid
        """
        return 0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0])

    def get_line_elem(self, direction, state):
        """
        returns a list of all elements in a straight line in a certain direction
        from the head of the snake
        """
        grid, score, alive, snake = state
        res = [0, 0, 0]  # food, snake, wall
        current = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        distance = 1  # Distance between the snake head and current position

        while self.is_in_grid(current, grid) and 0 in res:
            if FOOD_CHAR == grid[current[0]][current[1]]:
                res[0] = 1 / distance
            elif not res[1] and SNAKE_CHAR == grid[current[0]][current[1]]:
                res[1] = 1 / distance
            elif not res[2] and WALL_CHAR == grid[current[0]][current[1]]:
                res[2] = 1 / distance

            current = (current[0] + direction[0], current[1] + direction[1])
            distance += 1

        # For the border of the board (!= WALL_CHAR)
        if res[2] == 0:
            res[2] = 1 / distance

        return res

    def get_fitness(self):
        """
        Returns the fitness of the snake (float).
        """
        return self.fitness

    def get_nbr_move(self):
        """
        Returns the number of moves done by the snake from the beginning of the game (int)
        """
        return self.nbrMove

    def compute_fitness(self, gameScore):
        """
        Compute the fitness of the snake based on the number of moves done and its game score.
        Return the snake's fitness (float)
        """
        bonus = self.get_nbr_move()
        self.fitness = gameScore ** 2 * bonus

        return self.fitness

    def eat(self):
        """
        Increase the hunger of the snake. This hunger cannot exceed a certain value.
        """
        self.hunger += 75

        if self.hunger > 500:
            self.hunger = 500

    def reset_state(self):
        """
        Restore the hunger of the snake and reset its number of moves.
        """
        self.hunger = self.maxHunger
        self.nbrMove = 0
