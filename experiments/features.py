"""Feature blocks used by the experiments.

All blocks read the columns produced by common.add_columns, so training
and serving use the same preprocessing (backend.features.url_to_model_text).
"""

import ipaddress
import math
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def anchored_host(column):
    # "<" and ">" mark the start and end, so n-grams can see the first
    # label and the top-level domain as such
    return ["<" + host + ">" for host in column]


def _entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum(c / total * math.log2(c / total) for c in counts.values())


def _is_ip(host):
    try:
        ipaddress.ip_address(host)
        return 1.0
    except ValueError:
        return 0.0


def structural_matrix(frame):
    rows = []
    for host, rest in zip(frame["host"], frame["rest"]):
        labels = host.split(".")
        first = labels[0] if labels else ""
        digits = sum(c.isdigit() for c in host)
        rows.append([
            len(host),
            len(labels),
            host.count("-"),
            digits,
            digits / max(len(host), 1),
            max((len(label) for label in labels), default=0),
            len(first),
            _entropy(first),
            _is_ip(host),
            float(any(label.startswith("xn--") for label in labels)),
            float(not host.isascii()),
            len(rest),
            rest.count("/"),
            float("?" in rest),
            rest.count("="),
            rest.count("%"),
            float(len(rest) <= 1),
        ])
    return np.log1p(np.abs(np.array(rows, dtype=float)))


STRUCTURAL_NAMES = [
    "host_len", "host_labels", "host_hyphens", "host_digits", "host_digit_ratio",
    "longest_label", "first_label_len", "first_label_entropy", "is_ip", "punycode",
    "non_ascii_host", "rest_len", "rest_slashes", "has_query", "equals", "percent",
    "no_path",
]


class Structural(BaseEstimator, TransformerMixin):
    def fit(self, frame, y=None):
        return self

    def transform(self, frame):
        return structural_matrix(frame)


def text_block(ngram_range=(3, 5), max_features=100_000, min_df=3, sublinear_tf=False):
    return TfidfVectorizer(
        analyzer="char", ngram_range=ngram_range, min_df=min_df, lowercase=True,
        max_features=max_features, sublinear_tf=sublinear_tf, dtype=np.float32,
    )


def host_block(ngram_range=(2, 5), max_features=200_000, min_df=3, sublinear_tf=True):
    return make_pipeline(
        FunctionTransformer(anchored_host),
        TfidfVectorizer(
            analyzer="char", ngram_range=ngram_range, min_df=min_df, lowercase=True,
            max_features=max_features, sublinear_tf=sublinear_tf, dtype=np.float32,
        ),
    )


def combine(text=None, host=None, structural=False, structural_weight=1.0):
    blocks = []
    if text is not None:
        blocks.append(("text", text, "text"))
    if host is not None:
        blocks.append(("host", host, "host"))
    if structural:
        blocks.append(("structural", make_pipeline(Structural(), StandardScaler()),
                       ["host", "rest"]))
    weights = {"structural": structural_weight} if structural else None
    return ColumnTransformer(blocks, transformer_weights=weights)
