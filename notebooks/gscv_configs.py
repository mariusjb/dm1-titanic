from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
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
