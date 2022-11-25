from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


@dataclass
class GSCVConfig:
    name: str
    estimator: object
    param_grid: dict
    k_fold: StratifiedKFold


gscv_cfg_dt = GSCVConfig(
    "decision_tree",
    DecisionTreeClassifier(),
    {
        "criterion": ["log_loss"],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": ["sqrt"],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_rf = GSCVConfig(
    "random_forest",
    RandomForestClassifier(),
    {
        "n_estimators": [32, 64, 128],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [2, 3, 4],
        "min_samples_split": [0.1, 0.2, 0.3, 0.4],
        "min_samples_leaf": [0.05, 0.1, 0.15],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_gb = GSCVConfig(
    "gradient_boosting",
    GradientBoostingClassifier(),
    {
        "learning_rate": [0.25, 0.3, 0.35],
        "n_estimators": [23, 25, 27],
        "max_depth": [8],
        "max_features": ["log2"],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_mlp = GSCVConfig(
    "multilayer_perceptron",
    MLPClassifier(),
    {
        "hidden_layer_sizes": [(10,), (11,), (12,)],
        "activation": ["relu"],
        "solver": ["adam"],
        "batch_size": ["auto"],
        "learning_rate": ["invscaling"],
        "max_iter": [1400, 1500],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_knn = GSCVConfig(
    "knearest_neighbor",
    KNeighborsClassifier(),
    {
        "n_neighbors": [9, 10, 11],
        "weights": ["distance"],
        "algorithm": ["ball_tree"],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_sv = GSCVConfig(
    "support_vector",
    SVC(),
    {
        "gamma": ["scale", "auto"],
        "shrinking": [True, False],
        "class_weight": ["balanced", None],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_nb = GSCVConfig(
    "naive_bayes",
    GaussianNB(),
    {},
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_xgb = GSCVConfig(
    "xgboost",
    XGBClassifier(),
    {
        "learning_rate": [0.01],
        "max_depth": [8, 9],
        "colsample_bytree": [0.55, 0.6, 0.65],
        "n_estimators": [20, 25, 30],
        "gamma": [0.45, 0.5, 0.55],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)
