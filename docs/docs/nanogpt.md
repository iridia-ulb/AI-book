---
layout: post
title:  "NanoGPT"
nav_order: 14
categories: code
---

# NanoGPT

Vous trouverez ci-dessous les instructions et détails sur l'application de NanoGPT.
le but de cette application est de génerer automatiquement du texte "à la manière de".
Dans notre exemple le programme genère des textes ressemblants à des oeuvres de Shakespeare.

## Installation

Pour installer l'application, commencez par copier le dépot du livre ([AI-book sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```
git clone https://github.com/iridia-ulb/AI-book
```

Puis, accedez au dossier:

```bash
cd NanoGPT
```

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation

Pour lancer un entrainement avec le corpus de Shakespeare il suffit de lancer, par exemple, avec une `block_size` de 512 et 10 000 itérations:

```bash
poetry run python main.py --train --save_model my_new_model.pth --block_size 512 --eval_iters 10000
```

Il est aussi possible de changer le corpus d'entrainement avec le paramètre `--input` (par défaut celui-ci le programme prend le fichier `input.txt` contenant le corpus de Shakespeare).

Par la suite, une fois entrainé, pour tester votre modèle:

```bash
poetry run python main.py --infer --load_model my_new_model.pth --block_size 512
```

Veillez à bien sélectionner les mêmes paramètres lors de l'entrainement et de l'inférence (ici par exemple la `block_size`)

Toute une liste de paramètres supplémentaires sont configurables :
  
  * `--train` : Mode entraînement
  
  * `--infer` : Mode inférence
  
  * `--input` : fichier d'entrée de corpus d'entrainement
  
  * `--save_model SAVE_MODEL` : Sauvegarde le modèle dans le fichier spécifié
  
  * `--load_model LOAD_MODEL` : Charge le modèle depuis le fichier spécifié
  
  * `--batch_size BATCH_SIZE` : Nombre d'I/O que le modèle doit apprendre par batch
  
  * `--block_size BLOCK_SIZE` : Longueur des séquences que le transformer doit apprendre
  
  * `--max_iters MAX_ITERS` : Nombre d'itérations d'apprentissage
  
  * `--eval_interval EVAL_INTERVAL` : Intervalle d'évaluation pendant l'entraînement
  
  * `--learning_rate LEARNING_RATE` : Taux d'apprentissage
  
  * `--eval_iters EVAL_ITERS` : Nombre d'itérations d'évaluation
  
  * `--n_embd N_EMBD` : Dimension de l'espace dans lequel on projette les caractères
  
  * `--n_head N_HEAD` : Nombre de têtes d'attention
  
  * `--n_layer N_LAYER` : Nombre de couches
  
  * `--dropout DROPOUT` : Probabilité de dropout
  

En résumé:

```
usage: main.py [-h] [--train] [--infer] [--save_model SAVE_MODEL] [--load_model LOAD_MODEL] [--input INPUT] [--batch_size BATCH_SIZE]
               [--block_size BLOCK_SIZE] [--max_iters MAX_ITERS] [--eval_interval EVAL_INTERVAL] [--learning_rate LEARNING_RATE]
               [--eval_iters EVAL_ITERS] [--n_embd N_EMBD] [--n_head N_HEAD] [--n_layer N_LAYER] [--dropout DROPOUT]

Train and/or infer with a language model

options:
  -h, --help            show this help message and exit
  --train               Mode entraînement
  --infer               Mode inférence
  --save_model SAVE_MODEL
                        Sauvegarde le modèle dans le fichier spécifié
  --load_model LOAD_MODEL
                        Charge le modèle depuis le fichier spécifié
  --input INPUT         Utilise les données d'entrainement depuis le fichier spécifié
  --batch_size BATCH_SIZE
                        Nombre d'I/O que le modèle doit apprendre par batch
  --block_size BLOCK_SIZE
                        Longueur des séquences que le transformer doit apprendre
  --max_iters MAX_ITERS
                        Nombre d'itérations d'apprentissage
  --eval_interval EVAL_INTERVAL
                        Intervalle d'évaluation pendant l'entraînement
  --learning_rate LEARNING_RATE
                        Taux d'apprentissage
  --eval_iters EVAL_ITERS
                        Nombre d'itérations d'évaluation
  --n_embd N_EMBD       Dimension de l'espace dans lequel on projette les caractères
  --n_head N_HEAD       Nombre de têtes d'attention
  --n_layer N_LAYER     Nombre de couches
  --dropout DROPOUT     Probabilité de dropout
```
