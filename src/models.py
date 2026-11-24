"""Shared model helpers."""
import numpy as np
import pandas as pd


def expand_soft_labels(X, y):
    """Turn soft labels into weighted rows for classifiers that take one class per row.

    Each segment gets one row per class that received votes, weighted by that
    class's vote share, so every segment's weights add up to 1. Pass the result
    to fit(X_expanded, classes, sample_weight=weights).

    Expand only after splitting into folds, so a segment's rows stay together.

    Returns X_expanded, classes (class index per row), weights.
    """
    y = np.asarray(y)
    n_segments, n_classes = y.shape

    rows = np.repeat(np.arange(n_segments), n_classes)
    classes = np.tile(np.arange(n_classes), n_segments)
    weights = y.ravel()

    has_votes = weights > 0
    rows, classes, weights = rows[has_votes], classes[has_votes], weights[has_votes]

    if isinstance(X, pd.DataFrame):
        X_expanded = X.iloc[rows].reset_index(drop=True)
    else:
        X_expanded = np.asarray(X)[rows]
    return X_expanded, classes, weights
