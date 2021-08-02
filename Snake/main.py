import argparse
from gameModule import GUISnakeGame
from snakeTrainer import SnakesManager
from genetic_snake import Snake
from Astar_snake import IA_Astar
from dna import Dna
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Snake game.")
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
        help="Genetic algorithm: plays a move based of trained neural network, please select weight file",
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
        if args.astar or args.sshaped:
            agent = IA_Astar(args, game)
        elif args.genetic:
            with open(Path(args.genetic), "rb") as f:
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


if __name__ == "__main__":
    main()
