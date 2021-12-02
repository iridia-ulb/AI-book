---
layout: default
title: Foire aux questions (FAQ)
nav_order: 2.5
permalink: /faq
---

# FAQ
### Après avoir tapé poetry install, le terminal reste bloqué sur "Resolving dependencies..."

Il y a un problème avec le cache de poetry, pour vider le cache de poetry
vous devez supprimer les fichiers de caches grâce à la commande:

```
rm C:\Users\votrenom\AppData\Local\pypoetry\Cache\artifacts\*
```

en remplaçant `votrenom` par votre nom d'utilisateur.

puis relancer l'installation des dépendances avec poetry avec:

```
poetry install
```

### J'ai bien installé python mais je reçois une erreur lors du lancement du programme du type (windows):
```
Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases.
```

Vérifiez que vous lancer bien le programme avec la commande `python` et non `python3` 
dans la commande de lancement:

```
poetry run python main.py
```


### Je suis sous linux et le programme ne se lance pas correctement.

Essayer de remplacer `python` par `python3` dans la commande de lancement:
```
poetry run python3 main.py
```
