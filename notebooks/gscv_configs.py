from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


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
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [5, 25, 50, None],
        "min_samples_split": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", None],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_rf = GSCVConfig(
    "random_forest",
    RandomForestClassifier(),
    {
        "n_estimators": [10, 100, 200],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [5, 25, 50, None],
        "min_samples_split": [2, 3, 4, 5],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
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
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_mlp = GSCVConfig(
    "multilayer_perceptron",
    MLPClassifier(),
    {
        "hidden_layer_sizes": [(10,), (25,), (75,)],
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["lbfgs", "sgd", "adam"],
        "batch_size": ["auto"],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "max_iter": [1000],
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_knn = GSCVConfig(
    "knearest_neighbor",
    KNeighborsClassifier(),
    {
        "n_neighbors": [3, 5, 10],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_sv = GSCVConfig(
    "support_vector",
    SVC(),
    {
        "gamma": ["scale", "auto"],
        "shrinking": [True, False],
        "class_weight": ["balanced", None]
    },
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)

gscv_cfg_nb = GSCVConfig(
    "naive_bayes",
    GaussianNB(),
    {},
    StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
)
