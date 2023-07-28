---
layout: post
title:  "Spam Detector 2"
nav_order: 13
categories: code
---
# Detecteur de Spam (Spam detector)

Vous trouverez ci-dessous les instructions et détails sur l'application de détecteur de Spam (version 2).
Le but de cette application étant de determiner si un texte donné, venant typiquement d'un email
est catégorisé comme spam ou non.

L'application présenté ici utilise et compare un certain nombre d'algorithmes
de détection du spam (Bayes, MLP, Random Forest, ...)

## Installation

Pour installer l'application, commencez par copier le dépot du livre ([AI-book sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```
git clone https://github.com/iridia-ulb/AI-book
```

Puis, accedez au dossier:

```bash
cd SpamDetector2
```

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation

Vous pouvez ensuite lancer l'application dans l'un des modes:
show, train, test, classify ou compare.

Par exemple
```bash
poetry run python main.py show
```

En résumé:
```
usage: main.py [-h] {show,train,test,classify,compare} ...

Spam detector

positional arguments:
  {show,train,test,classify,compare}
                        Operation to run
    show                Show the most common spam words as a word cloud
    train               Train an extractor/classifier pair and save it
    test                Test an extractor/classifier pair and show metrics
    classify            Classify the given text as ham/spam using a specified extractor/classifier pair
    compare             Compare metrics of different extractor/classifier pairs

options:
  -h, --help            show this help message and exit
```

[ia-gh]: https://github.com/iridia-ulb/AI-book
