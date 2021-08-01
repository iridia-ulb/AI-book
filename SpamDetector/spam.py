from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log
import pandas as pd
import numpy as np

nltk.download("punkt")
nltk.download("stopwords")


def main():
    # Lecture du fichier spam.csv et transformation en DataFrame Pandas.
    mails = pd.read_csv("spam.csv", encoding="latin-1")
    # Supression des trois dernières colonnes
    mails.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    # On renomme les colonnes v1 et v2
    mails.rename(columns={"v1": "labels", "v2": "message"}, inplace=True)
    # On imprime le nombre de spam et de non spam
    print(mails["labels"].value_counts())
    # On rajoute une colonne "label" avec les valeurs: spam = 1, et non spam = 0
    mails["label"] = mails["labels"].map({"ham": 0, "spam": 1})
    # On supprime la colonne "labels"
    mails.drop(["labels"], axis=1, inplace=True)

    # On va maintenant réaliser les deux sous-ensembles que sont
    # Le training set (75% des données) et le Testing Set (25% des données).
    trainIndex, testIndex = list(), list()
    for i in range(mails.shape[0]):
        if np.random.uniform(0, 1) < 0.75:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = mails.loc[trainIndex]
    testData = mails.loc[testIndex]

    # On exécute un reset des index dans les deux sous-ensembles.
    trainData.reset_index(inplace=True)
    trainData.drop(["index"], axis=1, inplace=True)

    testData.reset_index(inplace=True)
    testData.drop(["index"], axis=1, inplace=True)

    # On peut visualiser les mots clés les plus fréquents des spams et
    # faire la même chose pour les non-spams.
    spam_words = " ".join(list(mails[mails["label"] == 1]["message"]))
    spam_wc = WordCloud(width=512, height=512).generate(spam_words)
    plt.figure(figsize=(10, 8), facecolor="k")
    plt.imshow(spam_wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    sc_tf_idf = SpamClassifier(trainData, "tf-idf")
    sc_tf_idf.train()
    preds_tf_idf = sc_tf_idf.predict(testData["message"])
    metrics(testData["label"], preds_tf_idf)

    sc_bow = SpamClassifier(trainData, "bow")
    sc_bow.train()
    preds_bow = sc_bow.predict(testData["message"])
    metrics(testData["label"], preds_bow)

    # prédit si un message est un spam ou pas.
    pm = process_message("I cant pick the phone right now. Pls send a message")
    print(sc_tf_idf.classify(pm))

    pm = process_message("Congratulations ur awarded $500 ")
    print(sc_tf_idf.classify(pm))


def process_message(message, lower_case=True, stem=True, stop_words=True, gram=1):
    """
    Cette fonction est très importante car c'est elle qui transforme les messages
    en une liste de mots clés essentiels: non stop et "stemmés".
    Si gram > 1 ce ne sont pas des mots clés mais des couples de mots clés qui sont
    pris en compte
    """
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [" ".join(words[i : i + gram])]
        return w
    if stop_words:
        sw = stopwords.words("english")
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


class SpamClassifier(object):
    def __init__(self, trainData, method="tf-idf"):
        self.mails, self.labels = trainData["message"], trainData["label"]
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == "tf-idf":
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_TF_and_IDF(self):
        noOfMessages = self.mails.shape[0]  # Nombre de messages
        self.spam_mails, self.ham_mails = (
            self.labels.value_counts()[1],
            self.labels.value_counts()[0],
        )
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0  # Nombre de mots dans les spams
        self.ham_words = 0  # Nombre de mots dans les non-spams
        self.tf_spam = dict()  # dictionnaire avec le TF de chaque mot dans les spam
        self.tf_ham = dict()  # dictionnaire avec le TF de chaque mot dans les non-spam
        self.idf_spam = dict()  # dictionnaire avec le IDF de chaque mot dans les spam
        self.idf_ham = (
            dict()
        )  # dicrionnaire avec le IDF de chaque mot dans les non-spam
        for i in range(noOfMessages):  # appelle les librairies nltk
            message_processed = process_message(self.mails.get(i))
            count = list()
            # Pour sauver si un mot est apparu dans le message ou non
            # IDF
            for word in message_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1  # calcule le TF d'un mot dans les spams
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1  # calcule le TF d'un mot dans les non spams
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                    # calcule le idf -> le nombre de spam qui contiennent ce mot
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1
                    # calcule le idf -> le nombre de non-spam qui contiennent ce mot

    def calc_prob(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        for (
            word
        ) in self.tf_spam:  # calcule la proba qu'un mot apparaisse dans les spams
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (
                self.spam_words + len(list(self.tf_spam.keys()))
            )
        for (
            word
        ) in self.tf_ham:  # calcule la proba qu'un mot apparaisse dans les non spams
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (
                self.ham_words + len(list(self.tf_ham.keys()))
            )
        self.prob_spam_mail, self.prob_ham_mail = (
            self.spam_mails / self.total_mails,
            self.ham_mails / self.total_mails,
        )

    def calc_TF_IDF(self):  # Effectue le calcul global avec le tf_idf.
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log(
                (self.spam_mails + self.ham_mails)
                / (self.idf_spam[word] + self.idf_ham.get(word, 0))
            )
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (
                self.sum_tf_idf_spam + len(list(self.prob_spam.keys()))
            )

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log(
                (self.spam_mails + self.ham_mails)
                / (self.idf_spam.get(word, 0) + self.idf_ham[word])
            )
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (
                self.sum_tf_idf_ham + len(list(self.prob_ham.keys()))
            )

        self.prob_spam_mail, self.prob_ham_mail = (
            self.spam_mails / self.total_mails,
            self.ham_mails / self.total_mails,
        )

    def classify(self, processed_message):  # classe les messages du test set
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                if self.method == "tf-idf":
                    pSpam -= log(
                        self.sum_tf_idf_spam + len(list(self.prob_spam.keys()))
                    )
                else:
                    pSpam -= log(self.spam_words + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                if self.method == "tf-idf":
                    pHam -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
                else:
                    pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam

    def predict(self, testData):  # Appelle le classifieur pour les messages du Test Set
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result


def metrics(labels, predictions):  # Calcule les métriques
    # True Positive, True Negative, False Positive, False Negative
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels.get(i) == 1 and predictions.get(i) == 1)
        true_neg += int(labels.get(i) == 0 and predictions.get(i) == 0)
        false_pos += int(labels.get(i) == 0 and predictions.get(i) == 1)
        false_neg += int(labels.get(i) == 1 and predictions.get(i) == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
