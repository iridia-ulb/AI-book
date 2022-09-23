---
layout: post
title:  "Tetris"
nav_order: 8
categories: code
---
# Tetris

Vous trouverez ci-dessous les instructions et détails sur le jeu du tetris.
Le but du jeu étant d'empiler le plus de pièce possibles pour former des lignes
complètes afin de les faire disparaitres et gagner des points.

Le jeu présenté ici utilise deux méthodes différentes:
le premier est un réseau de neurones artificiels entrainé grâce 
à un algorithme d'apprentissage par renforcement (reinforcement learning),
le deuxième est un controleur simple dont les paramètres sont optimisés
grâce à un algorithme génétique.

## Installation

Pour installer l'application, commencez par copier le dépot du livre ([AI-book sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```
git clone https://github.com/iridia-ulb/AI-book
```

Puis, accedez au dossier:

```bash
cd Tetris
```

Il y a ensuite deux sous-dossier, `TetrisRL` contient le programme fonctionnant
avec l'apprentissage par renforcement (RL) et `TetrisGA` contient le programme
fonctionnant avec l'algorithme génétique.
Rendez vous dans un de ces dossier avant de passer à l'étape suivante, par
exemple:

```bash
cd TetrisGA
```

Après avoir installé python et poetry, dans ce dossier, installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation de TetrisRL 
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

## Utilisation de TetrisGA 

Pour lancer le jeu avec uni controleur déjà entrainé:
```bash
poetry run python evaluation.py
```

Vous pouvez ajouter une option pour choisir un modèle pré-entrainé différent
de celui par défaut ("le dossier "SavedModel) avec l'option `-d`.

```bash
poetry run python evaluation.py -w temp_train/
```

Il est aussi possible de regler le nombre maximum de tetrominos 
avant l'arrêt du jeu avec l'option `-t`.

En résumé:
```bash
usage: evaluation.py [-h] [-d DIRECTORY] [-t TETROMINOES_LIMIT]

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

### Entrainement

Pour entrainer le modèle avec l'algorithme génétique, il suffit de lancer en
utilisant le script `training.py`

```bash
poetry run python training.py
```

Cette commande lancera l'interface pour configurer l'entrainement, vous pouvez y
choisir: les différent termes de l'heuristique à considérer, le nombre de
générations de l'entrainement, et la limite de temps pour chaque génération.

Une fois l'entrainement fini (ou annulé en quittant), un graphique s'affiche sur 
l'écran reprenant les données de la performance du modèle en fonction de la
génération.

Les résultats sont sauvegardés dans le dossier `temp_train/`

![tetrisGA screen](../assets/img/tetrisga.png)


[ia-gh]: https://github.com/iridia-ulb/AI-book
