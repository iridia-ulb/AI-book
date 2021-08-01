from gameModule import (
    RIGHT,
    LEFT,
    DOWN,
    UP,
    SNAKE_CHAR,
    FOOD_CHAR,
    WALL_CHAR,
    GUISnakeGame,
)
import argparse
import heapq
import time
from snakeManager import SnakesManager
from snake import Snake
from dna import Dna
import pickle


def main():
    parser = argparse.ArgumentParser(
        description="A* For Snake. If no argument is given, the game starts in player mode."
    )
    group_play = parser.add_mutually_exclusive_group(required=False)
    group_play.add_argument(
        "-p",
        "--player",
        action="store_true",
        help="Player mode: the player controls the game",
    )
    group_play.add_argument(
        "-x",
        "--ai",
        action="store_true",
        help="AI mode: the AI controls the game (requires an " "algorithm argument)",
    )
    group_play.add_argument(
        "-t",
        "--training",
        action="store_true",
        help="Training mode: the AI controls the game and a "
        "file is written to keep track of the scores ("
        "requires an algorithm argument and an output "
        "file)",
    )
    group_algorithm = parser.add_mutually_exclusive_group(required=False)
    group_algorithm.add_argument(
        "-g",
        "--genetic",
        action="store_true",
        help="Genetic algorithm: plays a move based of trained neural network",
    )
    group_algorithm.add_argument(
        "-s",
        "--sshaped",
        action="store_true",
        help="S-Shaped algorithm: browses the whole "
        "grid each time in an 'S' shape. Only "
        "works if height of grid is even.",
    )
    group_algorithm.add_argument(
        "-a",
        "--astar",
        action="store_true",
        help="A* algorithm: classical A* algorithm, with "
        "Manhattan distance as heuristic",
    )

    args = parser.parse_args()
    game = GUISnakeGame()
    game.init_pygame()

    agent = None
    if args.player:
        agent = None

    elif args.ai:
        if args.astar:
            agent = IA_Astar(args, game)
        elif args.genetic:
            with open(f"./weights/55.snake", "rb") as f:
                weights, bias = pickle.load(f)
            agent = Snake(Dna(weights, bias))

    elif args.training:
        population = 1000
        layers = [16, 16]
        mutation = 0.01
        hunger = 150
        elitism = 0.12
        snakesManager = SnakesManager(
            game,
            population,
            layersSize=layers,
            mutationRate=mutation,
            hunger=hunger,
            survivalProportion=elitism,
        )
        snakesManager.train()

    else:
        print("Please choose mode (-p,-x,-t)")

    print(agent)

    while game.is_running():
        game.next_tick(agent)

    game.cleanup_pygame()


class Node:
    def __init__(self, position, parent):
        self.position = position
        self.parent = parent

        self.g = 0
        self.h = 0
        self.f = self.g + self.h

    def __repr__(self):
        return f"({self.position[0]}, {self.position[1]})"

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.position == other.position
        else:
            return self.position == other


class IA_Astar:
    def __init__(self, args, game):
        self.moves = [RIGHT, DOWN, LEFT, UP]
        self.best_path = None  # The path used by the snake
        self.first = True  # Boolean
        self.args = args
        self.game = game

    def reset_state(self):
        self.best_path = None  # The path used by the snake
        self.first = True  # Boolean

    def eat(self):
        pass

    def choose_next_move(self, state):
        """
            This function is called by the game instance in order to find the next move chosen by the used algorithm.
            In our case, there are 5 different algorithms : Random, SShaped, A*, A* weighted and A* reversed. All of
            them are described in detail in the report

        :param state: The state containing the grid, the snake body, the score and a boolean indicating if the snake
                      is alive
        :return: The move chosen by the algorithm
        """

        grid, score, alive, snake = state
        head = snake[0]

        if self.args.astar:
            self.best_path = self.astar(state, self.game.food, interactive=False)
        elif self.args.sshaped:
            self.best_path = self.sshape(state)

        # the snake goes DOWN when A* does not work
        if self.best_path == 171:
            print("A* did not find path")
            return self.moves[1]

        next_move = self.get_next_move(self.best_path, head)

        return next_move

    def get_next_move(self, path, head):
        """
            This function finds the next move to do based on the head and the path.
        :param path: The path followed by the snake
        :param head: The head of the snake
        :return: The next move
        """
        next_node = path.pop()
        next_pos = next_node.position
        next_move = self.moves[0]

        next_mov_bool = []

        for i in range(len(next_pos)):
            next_mov_bool.append(next_pos[i] - head[i])

        if next_mov_bool[0] == 0:
            if next_mov_bool[1] == 1:
                next_move = self.moves[0]
            elif next_mov_bool[1] == -1:
                next_move = self.moves[2]
            else:
                print("Problem in moves, head: ", head, ", next_pos: ", next_pos)
        elif next_mov_bool[0] == 1:
            next_move = self.moves[1]
        elif next_mov_bool[0] == -1:
            next_move = self.moves[3]
        else:
            print("Problem in moves, head: ", head, ", next_pos: ", next_pos)

        return next_move

    def h_cost(self, current, end):
        """
            Cost used in the A* algorithm
        :param current: current node
        :param end: end node
        :return: the Manhattan distance between the current node and the end node
        """
        res = abs(current.position[0] - end.position[0]) + abs(
            current.position[1] - end.position[1]
        )
        return res

    def dist_to_snake(self, current, snake):
        """
            This function computes the minimum distance between a node and the snake body
        :param current: current node
        :param snake: snake body
        :return:
        """
        best_cost = 100000
        for i in snake:
            n = Node(i, None)
            cost = self.h_cost(current, n)
            if cost < best_cost:
                best_cost = cost
        return best_cost

    def astar(self, state, goal_pos, interactive=False):
        """
        This function is an implementation of the A* algorithm
        :param state: The current state of the game
        :param goal_pos: The position where the snake has to go
        :param interactive: Display the execution of the A* algorithm
        :return: The path to the goal
        """
        grid, score, alive, snake = state
        head = snake[0]
        closed_list = []
        open_list = []
        head_node = Node(head, None)
        food_node = Node(goal_pos, None)

        heapq.heappush(open_list, head_node)

        while open_list:
            current_node = heapq.heappop(open_list)
            closed_list.append(current_node)

            if interactive:
                time.sleep(0.1)
                self.game.grid[current_node.position[0]][current_node.position[1]] = "C"
                self.game.draw()

            if current_node.position == food_node.position:
                path = []
                while current_node.parent is not None:
                    path.append(current_node)
                    current_node = current_node.parent

                if interactive:
                    for el in path:
                        self.game.grid[el.position[0]][el.position[1]] = "A"
                        time.sleep(0.1)
                        self.game.draw()
                    for el in path:
                        self.game.grid[el.position[0]][el.position[1]] = " "
                    for el in open_list:
                        self.game.grid[el.position[0]][el.position[1]] = " "
                    for el in closed_list:
                        self.game.grid[el.position[0]][el.position[1]] = " "
                    self.game.grid[food_node.position[0]][
                        food_node.position[1]
                    ] = FOOD_CHAR
                    self.game.grid[head_node.position[0]][
                        head_node.position[1]
                    ] = SNAKE_CHAR
                    self.game.draw()

                return path

            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                node_position = (
                    current_node.position[0] + new_position[0],
                    current_node.position[1] + new_position[1],
                )

                # Make sure within range
                if (
                    node_position[0] > (len(grid) - 1)
                    or node_position[0] < 0
                    or node_position[1] > (len(grid[len(grid) - 1]) - 1)
                    or node_position[1] < 0
                ):
                    continue

                # Make sure walkable terrain
                if (
                    grid[node_position[0]][node_position[1]] == SNAKE_CHAR
                    or grid[node_position[0]][node_position[1]] == WALL_CHAR
                ):
                    continue

                # Create new node
                new_node = Node(node_position, current_node)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the closed list
                if child in closed_list:
                    continue

                if child in open_list:
                    if child.g > current_node.g + 1:
                        child.g = current_node.g + 1
                        child.parent = current_node

                else:
                    # Create the f, g, and h values
                    child.g = current_node.g + 1
                    child.h = self.h_cost(child, food_node)
                    child.f = child.g + child.h

                    child.parent = current_node

                    # Add the child to the open list
                    heapq.heappush(open_list, child)

                    if interactive:
                        if self.game.grid[child.position[0]][child.position[1]] == "+":
                            pass
                        else:
                            self.game.grid[child.position[0]][child.position[1]] = "S"
                        self.game.draw()
        return 171

    def sshape(self, state):
        """
            SShaped implementation. The snake does the same path during all the game and it gives the perfect score.
        :param state: The current state of the game
        :return: The path the snake has to follow
        """

        grid, score, alive, snake = state
        head = snake[0]
        path = []
        if score == 0:
            if head[0] == 0 and self.first:
                path.append(Node((head[0] + 1, head[1]), None))
                self.first = False
                return path

            else:
                self.first = False

            for i in range(head[1] + 1, len(grid[1])):
                path.append(Node((head[0], i), None))

            for i in range(1, head[0] + 1):
                path.append(Node((head[0] - i, head[1]), None))

        for i in range(len(grid)):
            if i % 2 == 1:
                for j in range(len(grid[0]) - 1):
                    path.append(Node((i, j), None))
            else:
                for j in range(len(grid[0]) - 2, -1, -1):
                    path.append(Node((i, j), None))

        for i in range(len(grid)):
            path.append(Node((len(grid) - i - 1, len(grid[0]) - 1), None))

        return path[::-1]


if __name__ == "__main__":
    main()
