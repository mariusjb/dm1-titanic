from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


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
        'class_weight': [None],
        'criterion': ['log_loss'],
        'max_features': [10],
        'min_impurity_decrease': [0.01],
        'min_samples_leaf': [9],
        'min_samples_split': [15],
        'splitter': ['best']
    },
    # {
    #     'class_weight': [None],
    #     'criterion': ['log_loss'],
    #     'max_features': [10],
    #     'min_impurity_decrease': [0.01],
    #     'min_samples_leaf': [9],
    #     'min_samples_split': [15],
    #     'splitter': ['best']
    # },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_rf = GSCVConfig(
    "random_forest",
    RandomForestClassifier(),
    {
        'bootstrap': [True, False], 
        'criterion': ['gini'], #, 'entropy', 'log_loss'],
        'max_depth': [2, 3, 4, 5, None], #10, None],
        'max_features': [None], #, 'sqrt', 'log2'],
        'min_samples_leaf': [1, 2, 3], #, 4, 5],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'n_estimators': [8, 10, 12, 14, 16, 32, 64, 128, 256, 512],
        # "bootstrap": [False],
        # "criterion": ["entropy"],
        # "max_depth": [None],
        # "max_features": ["log2"],
        # "min_samples_leaf": [2],
        # "min_samples_split": [5],
        # "n_estimators": [10]

        # "n_estimators": [32, 64, 128],
        # "criterion": ["gini", "entropy", "log_loss"],
        # "max_depth": [2, 3, 4],
        # "min_samples_split": [0.1, 0.2, 0.3, 0.4],
        # "min_samples_leaf": [0.05, 0.1, 0.15],
        # "max_features": ["sqrt", "log2", None],
        # "bootstrap": [True, False]
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

    # chiaras params
    # v1
    # {
    #     "learning_rate": [0.15],
    #     "subsample": [0.7],
    #     "max_depth": [1],
    #     "colsample_bylevel": [0.4],
    #     "colsample_bytree": [0.5],
    #     "n_estimators": [100],
    # },

    # v2
    # {
    #     "learning_rate": [0.15],
    #     "subsample": [0.9],
    #     "max_depth": [4],
    #     "colsample_bylevel": [0.4],
    #     "colsample_bytree": [0.15],
    #     "n_estimators": [30],
    #     "gamma": [1]
    # },

    # v3
    # {
    #     "learning_rate": [0.15],
    #     "subsample": [0.6],
    #     "max_depth": [3],
    #     "colsample_bylevel": [0.2],
    #     "colsample_bytree": [0.9],
    #     "n_estimators": [20],
    #     "gamma": [1]
    # },

    # # v4
    # {
    #     "learning_rate": [0.7],
    #     "subsample": [0.6],
    #     "max_depth": [3],
    #     "colsample_bylevel": [0.2],
    #     "colsample_bytree": [0.9],
    #     "n_estimators": [100],
    #     "gamma": [0]
    # },

    
    # my params
    {
        "learning_rate": [0.01],
        "max_depth": [8, 9],
        "colsample_bytree": [0.55, 0.6, 0.65],
        "n_estimators": [20, 25, 30],
        "gamma": [0.45, 0.5, 0.55],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_lr = GSCVConfig(
    "logistic_regression",
    LogisticRegression(),
    {},
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

