---
layout: post
title:  "Le plus court chemin"
nav_order: 4
categories: code
---
# Le plus court chemin (Shortest Path)

Vous trouverez ci-dessous les instructions et détails pour utiliser le programme de recherche
d'un plus court chemin. Le but du programme et de chercher un plus court chemin (shortest path)
dans un graph pondéré partiellement connecté.

On présente ici une technique classique de recherche avec l'algorithme de Dijkstra ainsi
qu'une recherche par A\*. La recherche se fait en premier lieu en partant du départ jusqu'à
l'arrivée, mais aussi en partant simultanément du départ et de l'arrivée pour se retrouver
et ainsi accelerer l'execution du programme.

## Installation

Pour installer l'application, commencez par copier le dépot du livre ([AI-book sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```
git clone https://github.com/iridia-ulb/AI-book
```

Puis, accedez au dossier:

```bash
cd Shortest_Path
```

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation

Vous pouvez ensuite lancer l'application, dans l'environnement virtuel
nouvellement crée, en utilsant la commande:

```bash
poetry run python main.py
```

Plusieurs options sont disponibles lors du lancement de la commande.
Il est par exemple possible de changer l'heuristique utilisée avec l'option
`--heuristic`, qui peut prendre les valeurs Manhattan, Euclidian, Chebyshev, ou
Dijkstra (si on veut utiliser cette algorithme à la place de A\*).
Il est aussi possible de choisir un fichier d'instance, qui permet de changer 
le graphe à parcourir, avec l'option `--instance` et d'y ajouter le nom du
fichier d'instance à ouvrir.
Pour finir il est possible de tester l'algorithme bidirectionnel.

Par exemple:
```bash
poetry run python main.py --heuristic Chebyshev --instance datasets/13_nodes.txt
```
permet de lancer l'instance `13_nodes.txt` avec l'heuristique de Chebyshev.

En résumé:

```
usage: main.py [-h] [--heuristic {Manhattan,Euclidian,Chebyshev,Dijkstra}] [--instance INSTANCE] [-b]
               [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Illustration of A* algorithm

optional arguments:
  -h, --help            show this help message and exit
  --heuristic {Manhattan,Euclidian,Chebyshev,Dijkstra}, --he {Manhattan,Euclidian,Chebyshev,Dijkstra}
                        Heuristic choice
  --instance INSTANCE   Path to instance
  -b, --bidirect        bidirectionnal
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logger level
```

NB: plusieurs instances sont disponibles dans le dossier `datasets`.


![path screen](../assets/img/shortest.png)

[ia-gh]: https://github.com/iridia-ulb/AI-book
