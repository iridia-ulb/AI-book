import argparse

from GUI_RunMenu import Menu, StartMenu

"""
Main file use to train the agents (from poetry commands)
"""


def main():
    parser = argparse.ArgumentParser(description="The Tetris game")
    parser.add_argument(
        "-t",
        "--time_limit",
        type=int,
        help="Maximum time for the entire training",
        default=float("inf"),
    )
    args = parser.parse_args()
    menu = StartMenu(args.time_limit, screen_width=600, screen_height=800, color_str="#000000")
    menu.run()


if __name__ == "__main__":
    main()

