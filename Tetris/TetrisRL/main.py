import sys
from agent import Agent
from tetris import Tetris
import argparse
from pathlib import Path


def main(render=True):
    parser = argparse.ArgumentParser(description="The Tetris game")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Path to weights file to load.",
        default="weights.h5",
    )

    args = parser.parse_args()
    if not Path(args.weights).is_file():
        parser.print_help()
        print()
        print(f"Invalid path for weight file: {args.weights}")
        sys.exit()

    # launch the environment
    game = Tetris()
    # initialize the agent
    agent = Agent(input_size=4)
    # load previous neural weights weights into the agents
    agent.load(Path(args.weights))

    running = True
    # while the game is not done, keep doing a step.
    while running:
        states = game.get_next_states()  # fetch all the next possible states.
        action, state = agent.act_best(
            states
        )  # the agent then decide the next action
        score, done = game.step(action, render=render)
        # inform the environment of the move and give the score and if the game ended.

        if done:  # if the game ended end the loop and print the score
            running = False
            print(
                f"End Score = {game.tetris_score} \nCleared Lines = {game.total_cleared_lines}"
            )


if __name__ == "__main__":
    main()
