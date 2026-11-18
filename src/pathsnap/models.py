"""Model zoo: the candidate classifiers compared in the experiments.

All are binary classifiers over (observation, candidate) rows that output
P(candidate is the reference arc). Seeds are fixed for reproducibility.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# name -> (family, factory, short config string, max training rows (None = all))
MODEL_SPECS = {
    "LogisticRegression": ("Linear", lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        "L2, C=1.0, standardised", None),
    "GaussianNB": ("Probabilistic", lambda: make_pipeline(StandardScaler(), GaussianNB()),
                   "var_smoothing=1e-9", None),
    "KNN": ("Instance-based", lambda: make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        "k=25, distance-weighted, standardised, 150k-row train sample", 150_000),
    "DecisionTree": ("Tree-based", lambda: DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=20, random_state=SEED),
        "max_depth=12, min_samples_leaf=20", None),
    "RandomForest": ("Ensemble (bagging)", lambda: RandomForestClassifier(
        n_estimators=150, max_depth=18, min_samples_leaf=5, max_samples=0.3,
        n_jobs=-1, random_state=SEED),
        "150 trees, max_depth=18, max_samples=0.3", None),
    "LightGBM": ("Boosting", lambda: LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1),
        "300 trees, lr=0.05, num_leaves=31 (untuned)", None),
    "MLP": ("Neural network", lambda: make_pipeline(
        StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                        max_iter=60, early_stopping=True, n_iter_no_change=8,
                                        random_state=SEED)),
        "64-32 ReLU, adam, early stopping", 400_000),
}
