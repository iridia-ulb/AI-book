# Le jeu du Taquin (8 Puzzle) avec apprentissage par renforcement (Reinforcement Learning) et recherche A\*
#### English below

## Installation

Après avoir installé python, pip et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

Vous pouvez ensuite lancé le jeu dans l'environnement virtuel nouvellement créé.
Le jeu en mode "recherche A\*" se lance comme ceci:
```bash
poetry run python3 8Puzzle_Astar.py
```

Le jeu en mode apprentissage par renforcement (Reinforcement Learning et plus spécifiquement 
Q learning) se lance comme ceci:
```bash
poetry run python3 8Puzzle_RL.py
```

Ensuite suivez les instructions à l'écran.

## Notes

Pour l'apprentissage par renforcement (Q learning) les "tables Q" (càd les IA déjà entrainées)
sont stockées dans le dossier `QTable` dans des fichiers texte (`QTable_#.txt`)

#### English version:
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

For the Q learning, the QTables (i.e. trained AI agents) are stored in the `QTable` folder
in text files (`QTable_#.txt`)
