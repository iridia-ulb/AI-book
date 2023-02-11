---
layout: post
title:  "Natural langage processing"
nav_order: 11
categories: code
---

# Natural langage processing

Vous trouverez ci-dessous les instructions et détails sur l'application
permettant .


## Installation

Pour installer l'application, commencez par copier le dépot du livre ([AI-book sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:

```
git clone https://github.com/iridia-ulb/AI-book
```

Puis, accedez au dossier :

```bash
cd nlp
```

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet :

```bash
poetry install
```

## Utilisation

Pour lancer le programme, utilisez la commande suivante:

```bash
poetry run python LDA.py
```

Le programme commencera par l'analyse Latent Dirichlet Allocation (LDA), et
affichera les nuages de mots des différentes catégories, puis executera les
analyses Word2Vec et Doc2Vec et affichera un graph des différents clusters 
détectés et des phrases analysés.

![LDA screenshot](../assets/img/lda.png)

[ia-gh]: https://github.com/iridia-ulb/AI-book
