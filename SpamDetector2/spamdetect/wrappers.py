from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


def nop(x):
  return x


class Doc2VecWrapper:
  def fit(self, X, y):
    docs = [TaggedDocument(d, [i]) for i, d in enumerate(X)]
    self.model = Doc2Vec(docs)
    return self

  def transform(self, X):
    vectors = [self.model.infer_vector(doc) for doc in X]
    return np.reshape(np.array(vectors), (len(X), self.model.vector_size))

  def fit_transform(self, X, y):
    self.fit(X, y)
    return self.transform(X)


class TfidfWrapper(TfidfVectorizer):
  def __init__(self, **kwargs):
    if 'preprocessor' not in kwargs:
      kwargs['preprocessor'] = nop
    if 'tokenizer' not in kwargs:
      kwargs['tokenizer'] = nop
    if 'token_pattern' not in kwargs:
      kwargs['token_pattern'] = None

    super().__init__(**kwargs)

  def fit(self, X, y):
    return super().fit(X, y)

  def transform(self, X):
    return super().transform(X).toarray()

  def fit_transform(self, X, y):
    return super().fit_transform(X, y)


class Switcher:
  """A switcher class to wrap multiple classifiers in a single object"""
  def __init__(self, n, Classifier, **kwargs):
    self.pre = KMeans(n, n_init='auto')
    self.classifiers = [Classifier(**kwargs) for _ in range(n)]

  def fit(self, X, y, **kwargs):
    self.pre.fit(X)
    C = self.pre.labels_

    # Fit sub-classifiers
    for c in range(len(self.classifiers)):
      self.classifiers[c].fit(X[C == c], y[C == c], **kwargs)

    return self

  def predict(self, X):
    C = self.pre.predict(X)

    # Predict with sub-classifiers
    pred = np.ndarray((len(X),), dtype=object)
    for c in range(len(self.classifiers)):
      if len(X[C == c]) > 0:
        pred[C == c] = self.classifiers[c].predict(X[C == c])

    return pred

  def score(self, X, y):
    C = self.pre.predict(X)

    # Score sub-classifiers
    scores = []
    for c in range(len(self.classifiers)):
      if len(X[C == c]) > 0:
        scores.append(self.classifiers[c].score(X[C == c], y[C == c]))

    return np.mean(scores)
