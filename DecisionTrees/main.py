# Import pandas library
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn import tree
from sklearn import ensemble
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns


def entropy_func(c, n):
    """
    The math formula
    """
    return -(c * 1.0 / n) * np.log2(c * 1.0 / n)


def entropy_cal(c1, c2):
    """
    Returns entropy of a group of data
    c1: count of one class
    c2: count of another class
    """
    if c1 == 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
        return 0
    return entropy_func(c1, c1 + c2) + entropy_func(c2, c1 + c2)


def entropy_of_one_division(division):
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:  # for each class, get entropy
        n_c = sum(division == c)
        e = n_c * 1.0 / n * entropy_cal(sum(division == c), sum(division != c))  # weighted avg
        s += e
    return s, n


# The whole entropy of two big circles combined
def get_entropy(y_predict, y_real):
    """
    Returns entropy of a split
    y_predict is the split decision, True/False, and y_true can be multi class
    """
    if len(y_predict) != len(y_real):
        print("They have to be the same length")
        return None
    n = len(y_real)
    s_true, n_true = entropy_of_one_division(y_real[y_predict])  # left hand side entropy
    s_false, n_false = entropy_of_one_division(y_real[~y_predict])  # right hand side entropy
    s = n_true * 1.0 / n * s_true + n_false * 1.0 / n * s_false  # overall entropy, again weighted average
    return s


def find_best_split(col, y):
    """
    col: the column we split on
    y: target var
    """
    min_entropy = 10
    for value in set(col):  # iterating through each value in the column
        y_predict = col < value  # separate y into 2 groups
        my_entropy = get_entropy(y_predict, y)  # get entropy of this split
        if my_entropy <= min_entropy:  # check if it's the best one so far
            min_entropy = my_entropy
            cutoff = value
    return min_entropy, cutoff


def find_best_split_of_all(x, y):
    """
    Find the best split from all features
    returns: the column to split on, the cutoff value, and the actual entropy
    """
    col = None
    min_entropy = 1
    cutoff = None
    for c in x.columns:  # iterating through each feature
        entropy, cur_cutoff = find_best_split(x[c], y)  # find the best split of that feature
        if entropy == 0:  # find the first perfect cutoff. Stop Iterating
            return c, cur_cutoff, entropy
        elif entropy <= min_entropy:  # check if it's best so far
            min_entropy = entropy
            col = c
            cutoff = cur_cutoff
    return col, cutoff, min_entropy


def fit(x, y, par_node={}, depth=0, max_depth=3):
    """
    x: Feature set
    y: target variable
    par_node: will be the tree generated for this x and y.
    depth: the depth of the current layer
    """
    y1 = y.tolist()
    if par_node is None:  # base case 1: tree stops at previous level
        return None
    elif len(y1) == 0:  # base case 2: no data in this group
        return None
    elif all_same(y1):  # base case 3: all y is the same in this group
        return {"val": y1[0]}
    elif depth >= max_depth:  # base case 4: max depth reached
        return None
    else:  # Recursively generate trees!
        # find one split given an information gain
        col, cutoff, entropy = find_best_split_of_all(x, y)

        df_left_idx = x[x[col] < cutoff].index
        df_left = x.loc[df_left_idx]
        df_left_y = y.loc[df_left_idx]

        df_right_idx = x[x[col] >= cutoff].index
        df_right = x.loc[df_right_idx]
        df_right_y = y.loc[df_right_idx]

        par_node = {
            "index_col": col,
            "cutoff": cutoff,
            "val": np.round(np.mean(y)),
        }  # save the information
        # generate tree for the left hand side data
        par_node["left"] = fit(df_left, df_left_y, {}, depth + 1, max_depth)
        # right hand side trees
        par_node["right"] = fit(df_right, df_right_y, {}, depth + 1, max_depth)
        depth += 1  # increase the depth since we call fit once
        return par_node


def all_same(items):
    return all(x == items[0] for x in items)


def predict(x, tree):
    return x.apply(lambda x: get_prediction(x, tree), axis=1)


def get_prediction(row, tree):
    cur_layer = tree  # get the tree we build in training
    while cur_layer.get("cutoff"):  # if not leaf node
        if row[cur_layer["index_col"]] < cur_layer["cutoff"]:
            if cur_layer["left"] == None:
                return cur_layer.get("val")
            cur_layer = cur_layer["left"]  # get the direction
        else:
            if cur_layer["right"] == None:
                return cur_layer.get("val")
            cur_layer = cur_layer["right"]
    else:  # if leaf node, return value
        return cur_layer.get("val")


def main():
    parser = argparse.ArgumentParser(description="Demonstration of decision trees")
    parser.add_argument("--max-depth", "-d", default=3, type=int)
    args = parser.parse_args()
    max_depth = args.max_depth

    # Creation et visualisation du DataFrame pandas
    df_train = pd.read_csv("HomeLoan_train.csv")
    # On drop toutes les données avec des valeurs manquantes
    df_train.dropna(axis=0, inplace=True)

    # On remplace toutes les données textuelles par des nombres
    df_train.Dependents.replace("3+", 3, inplace=True)
    df_train.Dependents = df_train.Dependents.astype("int")
    # On encode les données avec des catégories binaires sous forme de 0 ou de 1
    df_train.Gender = df_train.Gender.apply(lambda x: 0 if x == "Male" else 1)
    df_train.Married = df_train.Married.apply(lambda x: 0 if x == "No" else 1)
    df_train.Self_Employed = df_train.Self_Employed.apply(lambda x: 0 if x == "No" else 1)
    df_train.Education = df_train.Education.apply(lambda x: 0 if x == "Not Graduate" else 1)

    # On supprime les colonnes "Loan_ID" et "Property_Area"
    df_train.drop("Loan_ID", axis=1, inplace=True)
    df_train.drop("Property_Area", axis=1, inplace=True)

    # La variable cible est Loan Status
    # On encode la donnée cible à predire avec des catégories binaires sous forme de 0 ou de 1
    df_train.Loan_Status = df_train.Loan_Status.apply(lambda x: 0 if x == "N" else 1)

    print(df_train.describe())
    print(df_train.head())

    # On calcule la matrice de corrélation entre les données
    matrix = df_train.corr()
    f, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(matrix, vmax=0.8, square=True, cmap="BuPu", annot=True)
    plt.show()

    df_train_X = df_train.drop("Loan_Status", axis=1)
    df_train_y = df_train.Loan_Status

    X_train, X_test, y_train, y_test = train_test_split(df_train_X, df_train_y, test_size=0.3, random_state=42)

    tree1 = fit(X_train, y_train, max_depth=max_depth)
    print(X_test.describe())

    results1 = predict(X_test, tree1)
    score_tree1 = accuracy_score(results1, y_test) * 100
    print(f"Decision Tree1 accuracy: {score_tree1} %")

    # fit the decision tree classifier
    tree2 = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree2.fit(X_train, y_train)

    pred_cv_tree = tree2.predict(X_test)
    score_tree2 = accuracy_score(pred_cv_tree, y_test) * 100
    print(f"Decision Tree2 accuracy: {score_tree2} %")

    tree.plot_tree(tree2)
    plt.show()

    # fit the random forest classifier
    forest = ensemble.RandomForestClassifier(max_depth=max_depth)
    forest.fit(X_train, y_train)

    pred_cv_forest = forest.predict(X_test)
    score_forest = accuracy_score(pred_cv_forest, y_test) * 100
    print(f"Random Forest accuracy: {score_forest} %")


if __name__ == "__main__":
    main()
