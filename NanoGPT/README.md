# Documentation

1. Lancement du programme
2. Explication du code

<hr>

## 1. Lancement du programme

Entraîner le modèle :
```bash
py main.py --train --save_model model.pth
```

<br/>

Effectuer une inférence sur le modèle:
```bash
py main.py --infer --load_model model.pth
```

<br/>

Remplacez `model.pth` par le nom du fichier où vous voulez enregistrer/charger le modèle

<br/> 

Toute une liste de paramètres supplémentaires sont précisables :

  
  * `--train` : Mode entraînement
  
  * `--infer` : Mode inférence
  
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
  
  * `--head_size HEAD_SIZE` : Dimension des têtes d'attention dans laquelle sont projettés les caractères

<br/>

## 2. Explication du code

* _lignes 1 à 34 :_ <br>
    Cette première partie du code prépare les données nécessaires pour l'apprentissage du modèle.

<br>

### Classes

* _classe `BigramLanguageModel`_ : <br>
    Modèle de language basé sur des bigrammes utilisant un système de transformer.
   - `__init__()`: Initialise les composants du modèle (couches, blocs de transformer, tête de prédiction).
    - `forward()`: Effectue la propagation avant du modèle, calcule les logits des prédictions pour les caractères en fonction des entrées données.
    - `generate()`: Génère du texte à partir du modèle en prédisant itérativement le caractère suivant dans la séquence en fonction des caractères précédemment générés.

<br>

* _classe `Block`_ : <br>
    Bloc d'un modèle de transformer, composé de quatres couches.
    - `__init__()`: Initialise le bloc avec une couche d'attention multi-tête, une couche d'alimentation avant, et deux couches de normalisation.
    - `forward()`: Effectue la propagation avant du bloc en passant l'entrée à travers les deux premières couches.

<br>

* _classe `Head`_ : <br>
    Tête d'attention du modèle.
    - `__init__()`: Initialise la tête avec trois couches linéaires (clés, requêtes, valeurs).
    - `forward()`: Effectue la propagation avant de la tête en calculant les clés, requêtes et valeurs.

<br>

* _classe `FeedForward`_ : <br>
    Réseau de neurones à propagation avant utilisé dans chaque bloc du modèle.
    - `__init__()`: Initialise le réseau de neurones avec deux couches linéaires, une fonction d'activation ReLU et une couche de dropout.
    - `forward()`: Effectue la propagation avant du réseau en passant à travers chaque couche.

<br>

### Fonctions

* `get_batch()`: Obtient un batch de données pour l'entraînement en sélectionnant aléatoirement un offset et en séparant les données en entrées (x) et en cibles (y) de taille _block_size_.

* `estimate_loss()`: Calcule la perte moyenne sur un seul batch  pour les deux ensembles (entraînement et validation).

* `train()`: Entraîne le modèle sur un nombre maximal d'itérations donné en optimisant les paramètres du modèle.

* *`inference()`: Effectue l'inférence avec le modèle en générant du texte à partir d'un contexte initial.

* `save_model()`: Sauvegarde le modèle donné dans un fichier.

* `load_model()`: Charge un modèle à partir d'un fichier spécifié.

* `parse_args()`: Récupère les arguments spécifiés lors du lancement du script (cf 1. Lancement du programme).

<br>

### Modes

* **Mode entraînement (`--train`)** : initialise et entraîne le modèle tout en l'évaluant périodiquement pour surveiller ses performances afin d'éviter le surapprentissage. Pensez à enregistrer le modèle avec `--save_model` si vous voulez le réutiliser plus tard.

<br>

* **Mode inférence (`--infer`)** : génère du texte à partir du modèle entraîné puis l'affiche. Attention à bien lui préciser le modèle à charger avec `--load_model`.