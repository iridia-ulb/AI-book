import sys
import tetris
from agent import Agent
from tqdm import tqdm
import argparse
from pathlib import Path


def training(render=False):
    """
    The training of the agent with the tetris environment
    """
    parser = argparse.ArgumentParser(
        description="The Tetris game trainer for RL."
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Path to weights file to save to (default=weights.h5).",
        default="weights.h5",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        help="Number of episodes to train on (default=10000).",
        default=10000,
    )

    args = parser.parse_args()
    if Path(args.weights).is_file():
        parser.print_help()
        print()
        print(
            f"File {args.weights}, already exists, do you want to overwrite ?"
        )
        y = input("Type yes or no: ")
        if y != "yes":
            print("Aborting.")
            sys.exit()

    # --- Initialisation --- #
    game = tetris.Tetris(height=10)
    agent = Agent(input_size=4, decay=0.9995)
    saving_weights_each_steps = 1000
    print("\n >>> Begin Epsilon = " + str(agent.epsilon))
    print(" >>> Decay = " + str(agent.decay))

    # -- Episode LOOP -- #
    for i in tqdm(range(args.episodes)):
        # - Game and Board reset - #
        done = False
        game.reset()
        board = game.get_current_board_state()
        previous_state = game.get_state_properties(board)

        while not done:
            # fetch all the next possible states.
            possible_future_states = game.get_next_states()
            # the agent then decide the next action
            action, actual_state = agent.act_train(possible_future_states)

            # Performs the action
            reward, done = game.step(action, render=render)

            # Saves the move in memory
            agent.fill_memory(previous_state, actual_state, reward, done)

            # Resets iteration for the next move
            previous_state = actual_state

        # train the weights of the NN after the episode
        agent.training_montage()

        if i % saving_weights_each_steps == 0:
            agent.save(f"weights_temp_{i}.h5")
    agent.save(f"{args.weights}.h5")

    print("\n >>> End Epsilon = " + str(agent.epsilon))


if __name__ == "__main__":
    training()
