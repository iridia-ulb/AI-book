# Tetris

Vous trouverez ci-dessous les instructions et détails sur le jeu du tetris.
Le but du jeu étant d'empiler le plus de pièce possibles pour former des lignes
complètes afin de les faire disparaitres et gagner des points.

Le jeu présenté ici utilise un réseau de neurones artificiels entrainé grâce 
à un algorithme d'apprentissage par renforcement (reinforcement learning).

## Installation

Pour installer l'application, commencez par copier le dépot du livre ([AI-book sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```
git clone https://github.com/iridia-ulb/AI-book
```

Puis, accedez au dossier:

```bash
cd Tetris/TetrisRL
```

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation 
Pour lancer le jeu avec un réseau de neurones déjà entrainé:
```bash
poetry run python main.py
```

Vous pouvez ajouter une option pour choisir un modèle pré-entrainé différent
de celui par défaut ("weights.h5") avec l'option `-w`.

```bash
poetry run python main.py -w weights2.h5
```

Pour **quitter** le jeu, appuyez sur n'importe quelle touche dans la fenètre du
jeu, ou appuyez sur Ctrl+c dans le terminal.

En résumé:
```
usage: main.py [-h] [-w WEIGHTS]

The Tetris game

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHTS, --weights WEIGHTS
                        Path to weights file to load.
```

### Entrainement

Pour entrainer un nouveau réseau de neurones 
(Attention pour ce projet, il vous faudra probablement
un bon GPU pour espérer entrainer le réseau dans un temps acceptable) vous
pouvez utiliser le programme `train.py`:
```bash
poetry run python train.py -e 1000 -w weights2.h5
```
Ici l'option `-e` représente le nombre d'épisodes pendant lequel le réseau 
doit être entrainé, 10000 étant la valeur par défaut, et `-w` représente
le fichier dans lequel les poids synaptiques seront enregistrés à la fin de
l'entrainement.

En résumé:
```
usage: train.py [-h] [-w WEIGHTS] [-e EPISODES]

The Tetris game trainer for RL.

optional arguments:
  -h, --help            show this help message and exit
  -w WEIGHTS, --weights WEIGHTS
                        Path to weights file to save to (default=weights.h5).
  -e EPISODES, --episodes EPISODES
                        Number of episodes to train on (default=10000).
```

![tetris screen](../assets/img/tetris.png)

[ia-gh]: https://github.com/iridia-ulb/AI-book
