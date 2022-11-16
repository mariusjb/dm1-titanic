from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class GSCVConfig:
    name: str
    estimator: object
    param_grid: dict
    k_fold: KFold


gscv_cfg_dt = GSCVConfig(
    "decision_tree",
    DecisionTreeClassifier(),
    {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [5, 25, 50, None],
        "min_samples_split": np.arange(2, 5, 1),
        "min_samples_leaf": np.arange(1, 4, 1),
        "max_features": ["sqrt", "log2", None],
    },
    KFold(n_splits=10, shuffle=True),
)

gscv_cfg_rf = GSCVConfig(
    "random_forest",
    RandomForestClassifier(),
    {
        "n_estimators": [10, 100, 500, 1000],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [5, 25, 50, None],
        "min_samples_split": np.arange(2, 5, 1),
        "min_samples_leaf": np.arange(1, 4, 1),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    },
    KFold(n_splits=10, shuffle=True),
)

gscv_cfg_gb = GSCVConfig(
    "gradient_boosting",
    GradientBoostingClassifier(),
    {
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [10, 50, 100],
        "max_depth": [1, 3, 7],
        "max_features": ['sqrt', 'log2'],
    },
    KFold(n_splits=10, shuffle=True),
)

gscv_cfg_mlp = GSCVConfig(
    "multilayer_perceptron",
    MLPClassifier(),
    {
        "hidden_layer_sizes": [(50,), (100,), (500,)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "batch_size": ["auto"],
        "learning_rate": ["constant", "invscaling", "adaptive"],
    },
    KFold(n_splits=10, shuffle=True),
)
