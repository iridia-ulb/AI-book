# The 8-puzzle game using Reinforcement Learning and A\* search

## Installation

After having installed python and poetry, navigate to this folder and 
install the requirements of the project:

```bash
poetry install
```

Then launch the game inside the virtual environnement.
You can launch the game using the A\* AI using:
```bash
poetry run python3 8Puzzle_Astar.py
```

Or the game using reinforcement learning (specifically Q learning) using:
```bash
poetry run python3 8Puzzle_RL.py
```

And follow the instructions of screen.

## Notes

For the Q learning, the QTables (i.e. trained AI agents) are stored in the QTable folder
in text files (QTable_#.txt)
