"""Shared evaluation setup, so every model is compared on equal terms.

Everyone should use these metrics and these cross-validation folds.
"""
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42
N_SPLITS = 5
CLASS_NAMES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]


def kl_divergence(y_true, y_pred, eps=1e-7):
    """Mean KL(true || predicted) over segments. Lower is better, 0 is perfect."""
    y_pred = np.clip(y_pred, eps, 1)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return rel_entr(y_true, y_pred).sum(axis=1).mean()


def majority_class(y):
    """Index of the class with the most votes (ties go to the first class, like expert_consensus)."""
    return np.asarray(y).argmax(axis=1)


def macro_f1(y_true, y_pred):
    """Macro-F1 of the predicted top class against the majority vote. Higher is better."""
    return f1_score(majority_class(y_true), majority_class(y_pred),
                    labels=range(len(CLASS_NAMES)), average="macro", zero_division=0)


def evaluate(y_true, y_pred):
    """Both project metrics for one set of predictions."""
    return {"kl": kl_divergence(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred)}


def get_cv_folds(y, groups):
    """The shared 5 folds: each patient in one fold only, class balance kept similar across folds."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(y)), majority_class(y), groups))


def cross_validate(fit_predict, X, y, folds):
    """Score a model on the shared folds.

    fit_predict(X_fit, y_fit, X_val) must train on the first two and return
    predicted class probabilities for X_val, with columns in CLASS_NAMES order.
    """
    rows = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        y_pred = fit_predict(X.iloc[train_idx], y[train_idx], X.iloc[val_idx])
        rows.append({"fold": fold, **evaluate(y[val_idx], y_pred)})
    return pd.DataFrame(rows)
