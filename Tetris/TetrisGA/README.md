# Tetris with Genetic Algorithm

## C. Gullentops,  L. Quivron, V. Shulgach

You'll find below the instructions to launch our code.
Our goal is to train an agent to play tetris, using genetic algorithm.


## Installation 

You can find our code at the following address: https://github.com/LoicULB/TetrisGA

You should open a terminal in the main repository, and install
- python https://www.python.org/downloads/
- poetry https://python-poetry.org/docs/

Then, still in the main repository, you can enter:

```bash
poetry install
```

## Usage 

### Training

Our program allows you to train your own agents with the following command:

```bash
The Tetris game

optional arguments:
  -h, --help            show this help message and exit
  -t TIME_LIMIT, --time_limit TIME_LIMIT
                        Maximum time for the entire training
```

By default, the maximum training time is infinite.
You'll later be presented a menu in which you can choose more training parameters.


### Evaluation

The following command allows you to evaluate an agent performances trained using GA.
You can choose a source directory for the trained agent, the program will find the best agent from
the last generation of this directory.

```bash
The Tetris game

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        Path of saved generation on which to evaluate the best
                        agent
  -t TETROMINOES_LIMIT, --tetrominoes_limit TETROMINOES_LIMIT
                        The maximum number of tetrominoes after which the
                        evaluation stops
```

By default, we select one directory that we know is the best already trained agent.
The default number of tetrominoes is 500.

## Compare performance with deep-learning tetris

In the DQN_main.py file, we can see the modification that we made in order to be able to compare our agents versus the 
ones from DQN Tetris (we only put a Tetrominoes limit). If you want to do your evaluation : 
- Train the DQN with the desired amount of episodes (see their README)
- Train the GA agent with the desired Hyperparameters/Parameters (heuristic to consider, time limit for one generation)
  - put the number of generation sufficiently high for it to go the same amount of time as the DQN training
- Launch an evaluation on each project and choose a common tetrominoes limit


