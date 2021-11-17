import argparse

from gameModule import GUISnakeGame
from snakeTrainer import SnakesManager


def main():

    parser = argparse.ArgumentParser(
        description="Snake game, training program for neural net."
    )
    parser.add_argument(
        "-p",
        "--population",
        type=int,
        help="Defines the size of the initial population (must be >20), default=1000",
        default=1000,
    )
    parser.add_argument(
        "-m",
        "--mutation",
        type=float,
        help="Defines the mutation rate (0 < m < 1) (float), default=0.01",
        default=0.01,
    )
    parser.add_argument(
        "-e",
        "--elitism",
        type=float,
        help="Define the portion of snakes that are passed to next generation through elitism (0 < e < 1) (float), default=0.12",
        default=0.12,
    )
    args = parser.parse_args()

    game = GUISnakeGame()
    # game.init_pygame()

    population = args.population
    layers = [20, 10]
    mutation = args.mutation
    hunger = 150
    elitism = args.elitism
    snakesManager = SnakesManager(
        game,
        population,
        layersSize=layers,
        mutationRate=mutation,
        hunger=hunger,
        survivalProportion=elitism,
    )
    snakesManager.train()


if __name__ == "__main__":
    main()
