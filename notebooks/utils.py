import json
import logging
import os
from datetime import datetime

import dill
import pandas as pd
from sklearn.model_selection import GridSearchCV


def retrieve_latest_dataset():
    dir = f"../data"
    files = sorted([f for f in os.listdir(dir) if (f.endswith(".csv") and (f.startswith("preprocessed_2")))], reverse=True)
    latest = files[0]
    df = pd.read_csv(f"{dir}/{latest}", index_col=0)
    return df


def get_X_and_y(df: pd.DataFrame):
    return df.drop("Survived", axis=1), df["Survived"]


def run_grid_search_cv(gscv_dct, X_train, y_train):
    gscv = GridSearchCV(estimator=gscv_dct.estimator, param_grid=gscv_dct.param_grid, cv=gscv_dct.k_fold, verbose=2)
    gscv.fit(X_train, y_train)
    return gscv


def save_results_and_session(estimator_name: str, gscv_res: dict):
    iso_ts = datetime.now().isoformat(sep="T", timespec="seconds")

    # dump the results
    with open(f"../gscv_res/{estimator_name}/{str(iso_ts)}.json", "w") as fp:
        jsn = json.dumps({k: gscv_res[k] for k in gscv_res.keys()}, default=lambda x: str(x))
        fp.write(jsn)
    logging.warning(f"Results have been saved to '../gscv_res/{estimator_name}/{str(iso_ts)}.json'")

    # save the session
    dill.dump_session(f"sessions/{estimator_name}/{str(iso_ts)}.pkl")
    logging.warning(f"Session has been saved to '/sessions/{estimator_name}/{str(iso_ts)}.json'")


def load_latest_session(estimator_name: str):
    # include .pkl
    pkl_file = sorted([f for f in os.listdir(f"sessions/{estimator_name}/") if f.endswith(".pkl")])[-1]
    logging.warning(f"Session has been loaded from '/sessions/{estimator_name}/{pkl_file}'")
    return dill.load_session(f"sessions/{estimator_name}/{pkl_file}")  # Load the session
