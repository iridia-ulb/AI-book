import argparse
from pathlib import Path
import sys
import ast
import os
import pandas as pd
import re

from TetrisSolo import TetrisSolo
from TetrisAgents import GeneticAgent, TrainedAgent

"""
File used to evaluate the agent
"""


def main():
    parser = argparse.ArgumentParser(description="The Tetris game")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Path of saved generation on which to evaluate the best agent",
        default="./SavedModel/",  # TODO replace by our best trained agent
    )
    parser.add_argument(
        "-t",
        "--tetrominoes_limit",
        type=int,
        help="The maximum number of tetrominoes after which the evaluation stops",
        default=500,
    )

    args = parser.parse_args()
    if not Path(args.directory).is_dir():
        parser.print_help()
        print()
        print(f"Invalid path for generation file: {args.directory}")
        sys.exit()

    agent = retrieve_best_agent(args.directory)
    agent.weight_holes = agent.weight_array[0]
    agent.weight_height = agent.weight_array[1]
    agent.weight_bumpiness = agent.weight_array[2]
    agent.weight_line_clear = agent.weight_array[3]
    agent.weight_hollow_columns = agent.weight_array[4]
    agent.weight_row_transition = agent.weight_array[5]
    agent.weight_col_transition = agent.weight_array[6]
    agent.weight_pit_count = agent.weight_array[7]
    scores = []

    game = TetrisSolo(args.tetrominoes_limit, agent.weight_to_consider, agent)

    game.launch()
    scores.append(game.tetris_game.score)


"""
Used to retrieve the best agent from a training directory (directory of csv for each generation)
"""


def retrieve_max_file(directory_path: str):
    """
    Retrieve the file corresponding to the data of the last generation of a training directory
    :param directory_path: the training directory path
    """
    list_files = os.listdir(directory_path)
    pattern = re.compile("model_gen_[0-9]+.csv")
    max_nb_gen = 0
    for file in list_files:
        if pattern.match(file):
            nb_gen = int(re.search(r"\d+", file).group())
            if nb_gen > max_nb_gen:
                max_nb_gen = nb_gen
    if max_nb_gen == 0:
        raise "No valid file in this directory, must be regex : 'model_gen_number.csv'"

    return f"./{directory_path}/model_gen_{max_nb_gen}.csv"


def retrieve_best_agent(directory_path: str):
    """
    Return the best agent from the last generation of a training directory
    :param directory_path: the training directory
    """
    max_file = retrieve_max_file(directory_path)
    df = pd.read_csv(max_file)
    row_max_score = df["score"].argmax()
    agent_info = list(
        df.iloc[
            row_max_score,
        ]
    )

    ast_lit = ast.literal_eval(agent_info[8])
    return TrainedAgent(agent_info[0:8], ast_lit)


if __name__ == "__main__":
    main()

