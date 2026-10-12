"""Model builders used by the experiments."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline

INPUT_COLUMNS = ["text", "host", "rest", "url"]

BASELINE_TFIDF = dict(analyzer="char", ngram_range=(3, 5), min_df=3, lowercase=True)


def baseline_features():
    return ColumnTransformer([
        ("text", TfidfVectorizer(**BASELINE_TFIDF, max_features=100_000), "text"),
    ])


def baseline_classifier():
    return SGDClassifier(loss="log_loss", random_state=42, max_iter=1000)


def build(features=None, classifier=None):
    return make_pipeline(features or baseline_features(), classifier or baseline_classifier())


def fit(model, frame, sample_weight=None):
    step = model.steps[-1][0]
    params = {} if sample_weight is None else {f"{step}__sample_weight": sample_weight}
    model.fit(frame[INPUT_COLUMNS], frame["y"].to_numpy(), **params)
    return model


def scorer(model):
    column = list(model.classes_).index(0)

    def score(frame):
        return model.predict_proba(frame[INPUT_COLUMNS])[:, column]

    return score


def text_scorer(model):
    """For pipelines that take the model text directly (the saved model)."""
    column = list(model.classes_).index(0)

    def score(frame):
        return model.predict_proba(frame["text"])[:, column]

    return score


def logistic(C=1.0, class_weight=None):
    return LogisticRegression(C=C, solver="liblinear", max_iter=2000, class_weight=class_weight)


def sgd(alpha=1e-4, class_weight=None, random_state=42):
    return SGDClassifier(
        loss="log_loss", alpha=alpha, random_state=random_state, max_iter=1000,
        class_weight=class_weight,
    )


class Prefit:
    """A fitted feature transformer plus a fitted classifier."""

    def __init__(self, features, classifier):
        self.features = features
        self.classifier = classifier
        self.column = list(classifier.classes_).index(0)

    def score(self, frame):
        return self.classifier.predict_proba(self.features.transform(frame))[:, self.column]


def fit_matrix(classifier, X, y, sample_weight=None):
    if sample_weight is None:
        return classifier.fit(X, y)
    return classifier.fit(X, y, sample_weight=sample_weight)
