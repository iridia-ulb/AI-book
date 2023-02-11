import random
from collections import Counter
import matplotlib.pyplot as plt
import re
from re import RegexFlag
from wordcloud import WordCloud
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import numpy as np
from sklearn import cluster
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


class LDA(object):
    """
    Latent Dirichlet Allocation (LDA), a topic model designed for text documents.

    Topic models extract the key concepts in a set of documents.
    Each concept can be described by a list of keywords from most to least important.
    Then, each document can be connected to those concepts, or topics, to determine
    how representative that document is of that overall concept.
    """

    def __init__(self, K, max_iteration):
        self.K = K
        self.max_iteration = max_iteration

    def sample_from_weights(self, weights):
        """
        This function randomly choose an index based on an arbitrary set of weights.
        Return the first weight's index that is greater than or equal to a random number.
        """
        total = sum(weights)
        rnd = total * random.random()  # uniform between 0 and total
        for i, w in enumerate(weights):
            rnd -= w  # return the smallest i such that
            if rnd <= 0:
                return i  # sum(weights[:(i+1)]) >= rnd

    def p_topic_given_document(self, topic, d, alpha=0.1):
        """
        P(topic|d,Alpha)
        The fraction of words in document d
        that are assigned to topic (plus some smoothing)
        """
        return (self.document_topic_counts[d][topic] + alpha) / (self.document_lengths[d] + self.K * alpha)

    def p_word_given_topic(self, word, topic, beta=0.1):
        """
        P(word|topic,Beta)
        The fraction of words assigned to topic
        that equal word (plus some smoothing)
        """
        return (self.topic_word_counts[topic][word] + beta) / (self.topic_counts[topic] + self.W * beta)

    def topic_weight(self, d, word, topic):
        """
        P(topic|word,Alpha,Beta) = P(topic|d,Alpha) * P(word|topic,Beta)
        Given a document and a word in that document,
        return the weight of the word for the k-th topic in that document
        """
        return self.p_word_given_topic(word, topic) * self.p_topic_given_document(topic, d)

    def choose_new_topic(self, d, word):
        return self.sample_from_weights([self.topic_weight(d, word, k) for k in range(self.K)])

    def gibbs_sample(self, documents, document_topics):
        """
        Gibbs sampling https://en.wikipedia.org/wiki/Gibbs_sampling.
        """
        for _ in range(self.max_iteration):
            for d in range(self.D):
                for i, (word, topic) in enumerate(zip(documents[d], document_topics[d])):
                    # remove this word / topic from the counts
                    # so that it doesn't influence the weights
                    self.document_topic_counts[d][topic] -= 1
                    self.topic_word_counts[topic][word] -= 1
                    self.topic_counts[topic] -= 1
                    self.document_lengths[d] -= 1

                    # choose a new topic based on the weights
                    new_topic = self.choose_new_topic(d, word)
                    document_topics[d][i] = new_topic

                    # and now add it back to the counts
                    self.document_topic_counts[d][new_topic] += 1
                    self.topic_word_counts[new_topic][word] += 1
                    self.topic_counts[new_topic] += 1
                    self.document_lengths[d] += 1

    def run(self, documents):
        # How many times each topic is assigned to each document.
        self.document_topic_counts = [Counter() for _ in documents]
        # How many times each word is assigned to each topic.
        self.topic_word_counts = [Counter() for _ in range(self.K)]
        # The total number of words assigned to each topic.
        self.topic_counts = [0 for _ in range(self.K)]
        # The total number of words contained in each document.
        self.document_lengths = [len(d) for d in documents]
        self.distinct_words = set(word for document in documents for word in document)
        # The number of distinct words
        self.W = len(self.distinct_words)
        # The number of documents
        self.D = len(documents)
        # document_topics is a Collection that assign a topic (number between 0 and K-1) to each word in each document.
        # For example: document_topic[3][4] -> [4 document][id of topic assigned to 5 word]
        # This collection defines each document's distribution over topics, and
        # implicitly defines each topic's distribution over words.
        document_topics = [[random.randrange(self.K) for word in document] for document in documents]

        for d in range(self.D):
            for word, topic in zip(documents[d], document_topics[d]):
                self.document_topic_counts[d][topic] += 1
                self.topic_word_counts[topic][word] += 1
                self.topic_counts[topic] += 1

        self.gibbs_sample(documents, document_topics)

        return (self.topic_word_counts, self.document_topic_counts)

    def plot_words_clouds_topic(self, topic_names, plt):
        for topic in range(self.K):
            data = []
            text = ""
            for word, count in self.topic_word_counts[topic].most_common():
                if count > 1:
                    data.append(word)
            text = " ".join(data)
            # Generate a word cloud image
            wordcloud = WordCloud().generate(text)
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title("Topic #" + str(topic_names[topic]))
            plt.show()


def pre_process_documents(doc):
    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words("english")
    for i in range(len(doc)):
        # lower case and remove special characters\whitespaces
        doc[i][0] = re.sub(r"[^a-zA-Z\s]", "", doc[i][0], flags=RegexFlag.IGNORECASE | RegexFlag.A)
        doc[i][0] = doc[i][0].lower()
        doc[i][0] = doc[i][0].strip()
        # tokenize document
        tokens = wpt.tokenize(doc[i][0])
        # filter stopwords out of document
        filtered_tokens = [token for token in tokens if token not in stop_words]
        doc[i] = filtered_tokens

    # return filtered_tokens
    return doc


if __name__ == "__main__":
    random.seed(0)

    # Collection of documents.
    documents = [
        ["The sky is blue and beautiful."],
        ["Love this blue and beautiful sky!"],
        ["The quick brown fox jumps over the lazy dog."],
        ["A king's breakfast has sausages, ham, bacon, eggs, toast and beans"],
        ["I love green eggs, ham, sausages and bacon!"],
        ["The brown fox is quick and the blue dog is lazy!"],
        ["The sky is very blue and the sky is very beautiful today"],
        ["The dog is lazy but the brown fox is quick!"],
    ]

    topic_names = ["food", "weather", "animals"]

    pre_processed_documents = pre_process_documents(documents[:])
    print(pre_processed_documents)

    K = 3
    max_iteration = 1000

    lda = LDA(K, max_iteration)
    lda.run(pre_processed_documents)
    lda.plot_words_clouds_topic(topic_names, plt)

    sentences = [l[0].split() for l in documents]
    print(sentences)
    model = Word2Vec(sentences, min_count=1)
    print(model.wv.similarity("sky", "blue"))
    # print(model.wv.similarity("brown", "fox"))
    print(model.wv.similarity("dog", "lazy"))
    print(model.wv.most_similar(positive=["lazy"], negative=[], topn=2))

    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(sentences)]
    model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, epochs=100)
    dv_vectors = model.docvecs.vectors

    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(dv_vectors)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Cluster id labels for inputted data")
    print(labels)
    # print("Centroids data")
    # print(centroids)

    print(
        "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):"
    )
    print(kmeans.score(dv_vectors))

    silhouette_score = metrics.silhouette_score(dv_vectors, labels, metric="euclidean")

    print("Silhouette_score: ")
    print(silhouette_score)
    # Transform the data
    pca = PCA(2)
    df = pca.fit_transform(dv_vectors)
    label = kmeans.fit_predict(df)
    print(label)
    u_labels = np.unique(label)

    # plotting the results:
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.legend()
    plt.show()
