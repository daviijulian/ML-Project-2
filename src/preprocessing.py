"""Shared preprocessing and feature selection.

Everything is learned from the training data and then applied to other data,
so no test information leaks into the model.

Typical use:
    train, test = load_data()
    X_train, X_test = preprocess(train[get_feature_cols(train)], test[get_feature_cols(test)])
    features = load_selected_features()
    X_train, X_test = X_train[features], X_test[features]
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler

from . import VOTE_COLS

SKEW_THRESHOLD = 2
NEAR_CONSTANT_THRESHOLD = 0.95
CORRELATION_THRESHOLD = 0.95
N_FEATURES = 30  # smallest count on the KL plateau (see the notebook comparison)

FEATURES_PATH = Path(__file__).resolve().parent.parent / "results" / "selected_features.json"


def get_feature_cols(df):
    """Return the EEG and spectrogram feature columns of a data table."""
    return [col for col in df.columns if col.startswith(("eeg_", "spect_"))]


def get_soft_labels(df):
    """Return each segment's vote shares per class as an array."""
    return df[VOTE_COLS].div(df["num_votes"], axis=0).to_numpy()


class Preprocessor:
    """Median imputation, Yeo-Johnson on skewed features, then standardization."""

    def fit(self, X):
        self.imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
        X_imputed = self.imputer.fit_transform(X)

        self.skewed_cols = X_imputed.columns[X_imputed.skew().abs() > SKEW_THRESHOLD].tolist()
        self.power = PowerTransformer(method="yeo-johnson", standardize=False).set_output(transform="pandas")
        X_power = X_imputed.copy()
        if self.skewed_cols:
            X_power[self.skewed_cols] = self.power.fit_transform(X_imputed[self.skewed_cols])

        self.scaler = StandardScaler().set_output(transform="pandas")
        self.scaler.fit(X_power)
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        X_power = X_imputed.copy()
        if self.skewed_cols:
            X_power[self.skewed_cols] = self.power.transform(X_imputed[self.skewed_cols])
        return self.scaler.transform(X_power)


def preprocess(X_train, X_test):
    """Fit preprocessing on train and return the transformed train and test sets."""
    preprocessor = Preprocessor().fit(X_train)
    return preprocessor.transform(X_train), preprocessor.transform(X_test)


def drop_near_constant(X):
    """Return the columns where no single value dominates."""
    top_share = X.apply(lambda col: col.value_counts(normalize=True).iloc[0])
    return [col for col in X.columns if top_share[col] < NEAR_CONSTANT_THRESHOLD]


def drop_correlated(X):
    """Return columns, keeping only the first of any highly correlated group."""
    corr = X.corr().abs()
    kept = []
    for col in X.columns:
        if (corr.loc[col, kept] <= CORRELATION_THRESHOLD).all():
            kept.append(col)
    return kept


def rank_by_importance(X, y, groups, n_splits=5, random_state=42):
    """Rank features by random forest importance, averaged over patient-grouped folds."""
    cv = GroupKFold(n_splits=n_splits)
    fold_importances = []
    for train_idx, _ in cv.split(X, y, groups):
        forest = RandomForestRegressor(n_estimators=200, max_features="sqrt", min_samples_leaf=5,
                                       n_jobs=-1, random_state=random_state)
        forest.fit(X.iloc[train_idx], y[train_idx])
        fold_importances.append(forest.feature_importances_)
    importance = pd.Series(np.mean(fold_importances, axis=0), index=X.columns)
    return importance.sort_values(ascending=False)


def select_features(X_train_scaled, y_train, groups, n_features=N_FEATURES):
    """Run the full selection on preprocessed training data and return feature names."""
    candidates = drop_near_constant(X_train_scaled)
    kept = drop_correlated(X_train_scaled[candidates])
    importance = rank_by_importance(X_train_scaled[kept], y_train, groups)
    return importance.head(n_features).index.tolist()


def save_selected_features(features, path=FEATURES_PATH):
    """Save the chosen feature names so the whole team uses the same set."""
    with open(path, "w") as file:
        json.dump(features, file, indent=2)


def load_selected_features(path=FEATURES_PATH):
    """Load the team's chosen feature names."""
    with open(path) as file:
        return json.load(file)
