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
        self.brain = self.dna.get_model()
        self.fitness = None
        self.hunger = hunger
        self.maxHunger = hunger
        self.nbrMove = 0

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
        If the hunger of the snake is nul then return a string to indicate that the snakes is dead by starvation.
        vision (list): List containing distances between the head of the snake and other elements in the game
        Return the movement choice of the snake (tuple)
        """
        vision = self.get_simplified_state(state)
        if self.hunger > 0:
            self.hunger -= 1
            self.nbrMove += 1
            movesValues = self.brain.predict(vision)
            choice = 0

            movesValues = movesValues.tolist()

            # Chooses the best move (the move with the highest value)
            for i in range(1, len(movesValues[0])):
                if movesValues[0][i] > movesValues[0][choice]:
                    choice = i

            MOVEMENT = (RIGHT, LEFT, UP, DOWN)
            return MOVEMENT[choice]

        return "starve"

    def get_simplified_state(self, state):
        res = self.get_line_elem(RIGHT, state)
        res += self.get_line_elem((DOWN[0], RIGHT[1]), state)
        res += self.get_line_elem(DOWN, state)
        res += self.get_line_elem((DOWN[0], LEFT[1]), state)
        res += self.get_line_elem(LEFT, state)
        res += self.get_line_elem((UP[0], LEFT[1]), state)
        res += self.get_line_elem(UP, state)
        res += self.get_line_elem((UP[0], RIGHT[1]), state)

        # if self.previous_move == None:
        #     res += [0, 0]
        # elif self.pp_move == None:  # previous previous move
        #     res += [self.previous_move[0] / 2, self.previous_move[1] / 2]
        # else:
        #     dir = tupleAdd(self.previous_move, self.pp_move)
        #     res += [dir[0] / 2, dir[1] / 2]

        return res

    def is_in_grid(self, pos, grid):
        return 0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0])

    def get_line_elem(self, direction, state):
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

        if gameScore < 10:
            self.fitness = 2 ** gameScore * bonus ** 2
        else:
            self.fitness = 2 ** 10 * bonus ** 2 * (gameScore - 9)

        return self.fitness

    def get_hunger(self):
        """
        Returns the hunger of the snake (int).
        """
        return self.hunger

    def restore_food(self):
        """
        Set the hunger of the snake to the maximum value allowed.
        """
        self.hunger = self.maxHunger

    def eat(self):
        """
        Increase the hunger of the snake. This hunger cannot exceed a certain value.
        """
        self.hunger += 75

        if self.hunger > 500:
            self.hunger = 500

    def reset_nbr_move(self):
        """
        Set the number of moves done to 0.
        """
        self.nbrMove = 0

    def reset_state(self):
        """
        Restore the hunger of the snake and reset its number of moves.
        """
        self.restore_food()
        self.reset_nbr_move()


if __name__ == "__main__":
    NbrNodes = [rd.randint(10, 15) for i in range(rd.randint(1, 3))]
    dna1 = Dna(layersSize=NbrNodes)
    dna2 = Dna(layersSize=NbrNodes)
    snake1 = Snake(dna1)
    snake2 = Snake(dna2)
    snake1.mate(snake2)
