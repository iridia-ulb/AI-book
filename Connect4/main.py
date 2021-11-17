import pygame
import pygame.gfxdraw
import time
from connect4game import Connect4Game, Connect4Viewer
import time
from common import MONTE_CARLO, MINIMAX, RANDOM, SQUARE_SIZE
import sys
import argparse


def main():
    """
    One can run games of connect 4 between different types of bot and algorithms by giving as argument to the constructor
    Connect4Game the variables above.
    """
    parser = argparse.ArgumentParser(description="The Connect 4 game")
    parser.add_argument(
        "--player1",
        "--p1",
        "-1",
        type=str,
        help="Type of player for player 1",
        required=True,
        choices=("minimax", "mcts", "random", "human"),
        default="minimax",
    )
    parser.add_argument(
        "--player2",
        "--p2",
        "-2",
        type=str,
        help="Type of player for player 2",
        required=True,
        choices=("minimax", "mcts", "random"),
        default="MCTS",
    )
    args = parser.parse_args()

    nb_Games = 1
    total_games_won = [0, 0]
    # Change to True if one wants to play
    want_to_play = False
    p = [None, None]
    if args.player1 == "human":
        want_to_play = True
        p[0] = "Human"
    elif args.player1 == "random":
        p[0] = RANDOM
    elif args.player1 == "minimax":
        p[0] = MINIMAX
    elif args.player1 == "mcts":
        p[0] = MONTE_CARLO

    if args.player2 == "random":
        p[1] = RANDOM
    elif args.player2 == "minimax":
        p[1] = MINIMAX
    elif args.player2 == "mcts":
        p[1] = MONTE_CARLO

    game = Connect4Game(p[0], p[1], iteration=500, depth1=5, depth2=5)
    view = Connect4Viewer(game=game)
    view.initialize()

    running = True
    while running:
        if game._turn == 1 and game.get_win() is None and not want_to_play:
            game.bot_place()
        elif game._turn == -1 and game.get_win() is None:
            game.bot_place()
        elif game.get_win() is not None:
            if game.get_win() in (1, -1):
                total_games_won[(game.get_win() + 1) // 2] += 1
                running = False
            else:
                running = False

        for event in pygame.event.get():
            pygame.display.update()
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif (
                event.type == pygame.MOUSEBUTTONUP
                and event.button == 1
                and want_to_play
            ):
                game.place(pygame.mouse.get_pos()[0] // SQUARE_SIZE)

    if not want_to_play:
        print(
            f"Win rate of {game._player1}: {100 * (total_games_won[1] / nb_Games)}%"
        )
    else:
        print(
            f"You won with a win rate of: {100 * (total_games_won[1] / nb_Games)}%"
        )
    print(
        f"Win rate of {game._player2}: {100 * (total_games_won[0] / nb_Games)}%"
    )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


if __name__ == "__main__":
    main()
