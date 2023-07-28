from spamdetect import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from matplotlib import colors as mcolors
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import joblib as jl
import numpy as np
import argparse
import os
import time



EXTRACTOR = {
    'tfidf': TfidfWrapper,
    'doc2vec': Doc2VecWrapper,
}

SCALER = {
    'tfidf': MaxAbsScaler,
    'doc2vec': MinMaxScaler,
}

CLASSIFIERS = {
    'perceptron': Perceptron,
    'ridge': RidgeClassifier,
    'sgd': SGDClassifier,
    'bayes': ComplementNB,
    'knn': KNeighborsClassifier,
    'centroid': NearestCentroid,
    'ada': AdaBoostClassifier,
    'randomforest': RandomForestClassifier,
    'gboost': GradientBoostingClassifier,
    'tree': DecisionTreeClassifier,
    'svm': SVC,
    'mlp': MLPClassifier,
}

INIT_PARAMS = {
    'ada': {},
    'bayes': {},
    'centroid': {},
    'doc2vec': {},
    'gboost': {},
    'knn': {
        'n_neighbors': 5
    },
    'mlp': {},
    'perceptron': {},
    'randomforest': {},
    'ridge': {},
    'sgd': {},
    'svm': {},
    'tfidf': {},
    'tree': {},
}

FIT_PARAMS = {
    # Add 'extractor__' prefix to the following parameters
    'doc2vec': {},
    'tfidf': {},

    # Add 'classifier__' prefix to the following parameters
    'ada': {},
    'bayes': {},
    'centroid': {},
    'gboost': {},
    'knn': {},
    'mlp': {},
    'perceptron': {},
    'randomforest': {},
    'ridge': {},
    'sgd': {},
    'svm': {},
    'tree': {},
}

# Les noms de datasets doivent suivre la convention "{nom}.{train|test}.csv" avec ce split on prend le nécessaire
DATA_DIR = 'datasets'
DATASETS = {file.split('.',3)[1]: f"{DATA_DIR}/{file.replace('train', '%s').replace('test', '%s')}" for file in os.listdir(DATA_DIR) if '.csv' in file}
model_file = "models/%s-%s.%s.%d.plk"


def show(args):
    """Affiche les mots les plus fréquents des spams et des non-spams."""
    X, y = df_from_file(DATASETS[args.dataset] % 'train')
    spam_words = ' '.join(list(X[y == 'spam'].apply(lambda x: ' '.join(x))))
    spam_wc = WordCloud(width=512, height=512).generate(spam_words)
    plt.figure(figsize=(10, 8), facecolor="k")
    plt.imshow(spam_wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


def train(args):
    """Entraîne le modèle avec l'extracteur et le classifier donnés, sur le DataFrame donné."""
    X, y = df_from_file(DATASETS[args.dataset] % 'train')

    pipe = Pipeline([
        ('extractor', EXTRACTOR[args.extractor](**INIT_PARAMS[args.extractor])),
        ('scaler', SCALER[args.extractor]()),
        ('classifier', Switcher(args.preclassify, CLASSIFIERS[args.classifier], **INIT_PARAMS[args.classifier])),
    ])

    pipe.fit(X, y, **FIT_PARAMS[args.extractor], **FIT_PARAMS[args.classifier])

    # Enregistrer le modèle entrainé
    if not os.path.exists("models"):
        os.mkdir("models")
    jl.dump(pipe, args.model_file, compress=1)
    return pipe


def test(args):
    """Teste le modèle avec l'extracteur et le classifier données, sur le DataFrame donné."""
    pipe = check_for_model(args)
    X, y = df_from_file(DATASETS[args.dataset] % 'test')

    print(f"Results for {args.extractor} extractor with {args.classifier} classifier:")
    score = pipe.score(X, y)
    pred = pipe.predict(X)

    # Print des résultats
    print(f"Sklearn score: {score}")
    metrics = print_metrics(y.values, pred)
    print()

    return metrics


def classify(args):
    """Classifie le texte donné en paramètre."""
    pipe = check_for_model(args)

    if args.text:
        pm = process_message(args.text)
    elif args.file:
        with open(args.file, 'r') as f:
            pm = process_message(f.read())

    pred = pipe.predict([pm])
    print("Class:", pred)


def compare(args):
    """Compare les modèles spécifiés dans le fichier donné."""
    try:
        models = parse_models_file(args)
        n_metrics = 4
        n_models = len(models)
        results = np.zeros((n_models,n_metrics))
        timings = np.zeros(n_models)
        model_names = [f"{models[i][0]}-{models[i][1]}-{models[i][2]}" for i in range(n_models)]
        for i in range(n_models):
            start = time.time()
            args.extractor = models[i][0]
            args.classifier = models[i][1]
            args.preclassify = models[i][2]
            args.model_file = model_file % (args.dataset, args.extractor, args.classifier, args.preclassify)
            res = test(args)
            end = time.time()
            results[i] = res
            timings[i] = end-start
        plot_results(results.T, timings, model_names, args.file[0], args.dataset)
    except KeyError:
        print("Erroneous model in file, exiting.")
    except IndexError:
        print("Model comparison file format not respected, exiting.")


def check_for_model(args):
    """Vérifie qu'une version sauvegardée du modèle existe, en entraîne un nouveau sinon."""
    if not os.path.exists(args.model_file):
        print(f"No trained model for the chosen extractor {args.extractor} and classifier {args.classifier} with {args.preclassify} categories found, training one...")
        pipe = train(args)
    else:
        pipe = jl.load(args.model_file)

    return pipe


def df_from_file(file):
    """Retourne un DataFrame Pandas à partir d'un fichier csv."""
    df = pd.read_csv(file)
    df['text'] = df['text'].apply(process_message)
    return df['text'], df['label']


def parse_models_file(args):
    """Parse un fichier de comparaison de modèles."""
    models = []
    with open(args.file[0], "r") as file:
        for line in file:
            model_txt = line.split(' ')
            extractor = model_txt[0]
            classifier = model_txt[1].split('\n')[0]
            if len(model_txt) > 2:  # Si n_cat est spécifié
                n_cat = int(model_txt[2].split('\n')[0])
            else:
                n_cat = 1
            models.append((extractor, classifier, n_cat))
    return models


def plot_results(results, timings, model_names, fig_filename, dataset):
    """Fait un graphe comparant le sperformances des modèles à comparer."""
    labels = ("Precision", "Recall", "F-score", "Accuracy")
    colours = list(mcolors.TABLEAU_COLORS.values())
    n_models = len(model_names)
    rows = 2
    columns = 3
    fig, axs = plt.subplots(rows,columns, figsize=(20,13.33))
    # Graphe des mesures
    for i in range(len(results)):
        idx = (i//columns, i%columns)
        bar = axs[idx].bar([j for j in range(n_models)], results[i], color=colours[0:n_models])
        axs[idx].bar_label(bar, fmt='%.2f',padding=1.5)
        # Labels abscisse
        axs[idx].set_xticks([j for j in range(n_models)], model_names, weight='bold', fontsize=6)
        axs[idx].tick_params(axis='x',rotation=45) # Pivoter les labels de l'axe des abscisses (pour ne pas qu'ils se chevauchent)
        # Labels ordonnées
        axs[idx].set_ylabel(labels[i], fontweight='bold')

        axs[idx].plot

    # Légende basée sur graphe en haut à droite
    axs[0,columns-1].legend(axs[0, columns-1].containers[0], model_names, bbox_to_anchor=(1, 1), loc='upper left')

    # Graphe accélération
    idx_gacc = (rows-1,columns-1)
    # Normalization, neutralisation (pour avoir les proportions) et opposition (+ de temps = accélération négative), on garde tout les éléments auf le dernier (valant 0)
    accel_fct = ((timings/timings[-1]-1)*(-100))[:-1]
    bar = axs[idx_gacc].bar([j for j in range(n_models-1)],accel_fct, color=colours[0:(n_models-1)])
    axs[idx_gacc].bar_label(bar, fmt='%.2f%%',padding=2.5)
    # Labels abscisse
    axs[idx_gacc].set_xticks([j for j in range(n_models-1)], model_names[:-1], weight='bold', fontsize=6)
    axs[idx_gacc].tick_params(axis='x',rotation=45) # Pivoter les labels de l'axe des abscisses (pour ne pas qu'ils se chevauchent)
    # Labels ordonnée
    y_tick_locs = axs[idx_gacc].get_yticks()
    axs[idx_gacc].yaxis.set_major_locator(mticker.FixedLocator(y_tick_locs)) # Si on formatte sans ça on a un warning
    axs[idx_gacc].set_yticklabels([f"{fct}%" for fct in y_tick_locs])
    axs[idx_gacc].set_ylabel(f"Acceleration factor in proportion to {model_names[-1]}", fontweight='bold')

    axs[idx_gacc].plot

    axs[1,1].axis('off')
    fig.suptitle(f"Metrics and acceleration factor for {fig_filename} on dataset {dataset}", fontweight='bold', fontsize=16)
    plt.tight_layout()
    if not os.path.exists("figures"):
        os.mkdir("figures")
    plt.savefig(f"figures/{fig_filename.split('/')[-1].rsplit('.',1)[0]}_{dataset}.png")  # Rajout du nom du modèle 'principal' et du dataset, car peuvent varier pour plusieurs runs avec même fichier


def argv():
    """Parse les arguments passés au programme et retourne un objet contenant les arguments."""
    main_parser = argparse.ArgumentParser(description="Spam detector")
    subparsers = main_parser.add_subparsers(dest='command', help='Operation to run', required=True)

    # Parser pour l'opération 'show'
    show_parser = subparsers.add_parser('show', help="Show the most common spam words as a word cloud")
    show_parser.set_defaults(func=show)
    show_parser.add_argument('-d', '--dataset', type=str, nargs=1, choices=DATASETS, default=['sms'], help="The dataset to use for training")

    # Parser pour l'opération 'train'
    train_parser = subparsers.add_parser('train', help="Train an extractor/classifier pair and save it")
    train_parser.set_defaults(func=train)
    train_parser.add_argument('-e', '--extractor', type=str, nargs=1, choices=EXTRACTOR.keys(), default=['tfidf'], help="The feature extractor to use")
    train_parser.add_argument('-c', '--classifier', type=str, nargs=1, choices=CLASSIFIERS.keys(), default=['bayes'], help="The classifier to use")
    train_parser.add_argument('-d', '--dataset', type=str, nargs=1, choices=DATASETS, default=['sms'], help="The dataset to use for training")
    train_parser.add_argument('-p', '--preclassify', type=int, nargs=1, default=[1], help="""The number of categories to pre-classify the data into.
                                                                                            Default is 1, which means no pre-classification.""")

    # Parser pour l'opération 'test'
    test_parser = subparsers.add_parser('test', help="Test an extractor/classifier pair and show metrics")
    test_parser.set_defaults(func=test)
    test_parser.add_argument('-e', '--extractor', type=str, nargs=1, choices=EXTRACTOR.keys(), default=['tfidf'], help="The feature extractor to use")
    test_parser.add_argument('-c', '--classifier', type=str, nargs=1, choices=CLASSIFIERS.keys(), default=['bayes'], help="The classifier to use")
    test_parser.add_argument('-d', '--dataset', type=str, nargs=1, choices=DATASETS, default=['sms'], help="The dataset to use for testing")
    test_parser.add_argument('-p', '--preclassify', type=int, nargs=1, default=[1], help="""The number of categories to pre-classify the data into.
                                                                                            Default is 1, which means no pre-classification.""")

    # Parser pour l'opération 'classify'
    classify_parser = subparsers.add_parser('classify', help="Classify the given text as ham/spam using a specified extractor/classifier pair")
    classify_parser.set_defaults(func=classify)
    classify_parser.add_argument('-e', '--extractor', type=str, nargs=1, choices=EXTRACTOR.keys(), default=['tfidf'], help="The feature extractor to use")
    classify_parser.add_argument('-c', '--classifier', type=str, nargs=1, choices=CLASSIFIERS.keys(), default=['bayes'], help="The classifier to use")
    classify_parser.add_argument('-d', '--dataset', type=str, nargs=1, choices=DATASETS, default=['sms'], help="The dataset to use for classification")
    classify_parser.add_argument('-p', '--preclassify', type=int, nargs=1, default=[1], help="""The number of categories to pre-classify the data into.
                                                                                            Default is 1, which means no pre-classification.""")
    classify_source_group = classify_parser.add_mutually_exclusive_group(required=True)
    classify_source_group.add_argument('-t', '--text', help="The text to classify")
    classify_source_group.add_argument('-f', '--file', help="A file containing the text to classify")

    # Parser pour l'opération 'compare'
    compare_parser = subparsers.add_parser('compare', help="Compare metrics of different extractor/classifier pairs")
    compare_parser.set_defaults(func=compare)
    compare_parser.add_argument('-d', '--dataset', type=str, nargs=1, choices=DATASETS, default=['sms'], help="The dataset to use for classification")
    compare_parser.add_argument('-f', '--file', type=str, nargs=1, required=True, help="""The file to use for comparison.
                That file is a simple text file containing one or more sets
                of extractor and classifier. Each set needs to be in a new line with the extractor,
                the classifier, and then the number of categories, separated by a whitespace. A
                performances' plot is saved in the './figures' directory.""")

    args = main_parser.parse_args()

    if args.command in {'show', 'train', 'test', 'classify', 'compare'}:
        args.dataset = args.dataset[0]
    if args.command in {'train', 'test', 'classify'}:
        args.extractor = args.extractor[0]
        args.classifier = args.classifier[0]
        args.preclassify = args.preclassify[0]
        args.model_file = model_file % (args.dataset, args.extractor, args.classifier, args.preclassify)

    return args


def main():
    args = argv()
    args.func(args)


if __name__ == "__main__":
    main()
