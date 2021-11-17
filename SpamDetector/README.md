# Detecteur de Spam (Spam detector)

Vous trouverez ci-dessous les instructions et détails sur l'application de détecteur de Spam.
Le but de cette application étant de determiner si un texte donné, venant typiquement d'un email
est catégorisé comme spam ou non.

L'application présenté ici utilise un algorithme d'apprentissage du type 
classification naïve bayésienne (Naive Bayes classifier).

## Installation

Pour installer l'application, commencez par copier le dépot du livre ([AI-book sur github][ia-gh]),
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```
git clone https://github.com/iridia-ulb/AI-book
```

Puis, accedez au dossier:

```bash
cd SpamDetector
```

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation

Vous pouvez ensuite lancer l'application dans un des trois modes:
le mode `show` (`-s`) vous permettra de visualiser un nuage de mots des spams
contenus dans le fichier spam.csv;

```bash
poetry run python main.py -s
```

le mode `test` (`-t`) vous donnera les métriques de résultats d'un test de
classification de messages aléatoires après entrainement;

```bash
poetry run python main.py -t
```
le mode `classify` (`-c`) vous permettre de tester une phrase pour savoir si
elle sera détectée comme spam, sur l'algorithme, entrainé avec les données 
du fichier `spam.csv`.

```bash
poetry run python main.py -c "Can machines think?"
```

Vous verez alors apparaitre dans le terminal la mention
`Spam? : True`, ou `Spam? : False`, suivant si votre message est classé comme
indésirable ou non.

En résumé:
```
usage: main.py [-h] [-s] [-t] [-c CLASSIFY]

Spam detector.

optional arguments:
  -h, --help            show this help message and exit
  -s, --show            Shows the occurence of words as a wordcloud
  -t, --test            trains and tests the algorithms and gives results in différent metrics
  -c CLASSIFY, --classify CLASSIFY
                        Classifies the given text into spam or not spam using TFxIDF
```

![spam screen](../assets/img/spam.png)

[ia-gh]: https://github.com/iridia-ulb/AI-book
