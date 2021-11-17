import pygame
from Game_UI import WIDTH, HEIGHT, SlidePuzzle
import os
from EightPuzzle_astar import playAIGame
from EightPuzzle_RL import initPlayerAI
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="8Puzzle game.")

    parser.add_argument(
        "-a",
        "--astar",
        action="store_true",
        help="Start the program in A* mode.",
    )
    parser.add_argument(
        "-r", "--rl", action="store_true", help="Start the program in RL mode."
    )
    args = parser.parse_args()
    pygame.init()
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.display.set_caption("8-Puzzle game")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    fpsclock = pygame.time.Clock()

    while True:
        puzzle = SlidePuzzle((3, 3), 160, 5, screen)
        if args.astar:
            choice = puzzle.selectPlayerMenu("8 Puzzle using A* search")
        elif args.rl:
            choice = puzzle.selectPlayerMenu(
                "8 Puzzle using Reinforcement Learning"
            )
        else:
            parser.print_help()
            print()
            print("Please select an option (--astar, or --rl)")
            sys.exit()

        if choice == "AI" and args.astar:
            puzzle.shuffle()
            playAIGame(puzzle, fpsclock)
        elif choice == "AI" and args.rl:
            modelAI = puzzle.selectModel()
            if modelAI != "":
                AI = initPlayerAI(puzzle, modelAI)

                trainAI = puzzle.playTrainMenu()
                if trainAI:
                    trainingNb = puzzle.selectTrainingNb()
                    if trainingNb != 0:
                        AI.trainingAI(trainingNb)
                elif trainAI == False:
                    # puzzle.selectBoard(fpsclock)
                    puzzle.shuffle()
                    AI.playAIGame(fpsclock)
        else:
            puzzle.playHumanGame(fpsclock)


if __name__ == "__main__":
    main()
