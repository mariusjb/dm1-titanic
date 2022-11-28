import json
import logging
import os
from datetime import datetime
from typing import List, Union

import dill
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

def retrieve_latest_train_test(drop_col: List[str] = []) -> Union[pd.DataFrame, pd.DataFrame]:
    dir = f"../data"

    def read_data_of_type(type_data: str):
        files = sorted([f for f in os.listdir(dir) if (f.endswith(".csv") and (f.startswith(type_data)))], reverse=True)
        latest = files[0]
        df = pd.read_csv(f"{dir}/{latest}", index_col=0)
        return df.drop(drop_col, axis=1)

    return read_data_of_type("train_data"), read_data_of_type("test_data")


def run_grid_search_cv(gscv_dct, X_train, y_train, X_test):
    gscv = GridSearchCV(
        estimator=gscv_dct.estimator, param_grid=gscv_dct.param_grid, scoring="f1", error_score=0, cv=gscv_dct.k_fold, verbose=2
    )
    gscv.fit(X_train, y_train)
    pred = gscv.best_estimator_.predict(X_test) 
    return gscv, pred



def save_results_and_session(estimator_name: str, gscv: dict):
    iso_ts = datetime.now().isoformat(sep="T", timespec="seconds")

    # dump the results
    with open(f"../gscv/{estimator_name}/{str(iso_ts)}.json", "w") as fp:
        jsn = json.dumps({k: gscv[k] for k in gscv.keys()}, default=lambda x: str(x))
        fp.write(jsn)
    logging.warning(f"Results have been saved to '../gscv/{estimator_name}/{str(iso_ts)}.json'")

    # save the session
    dill.dump_session(f"sessions/{estimator_name}/{str(iso_ts)}.pkl")
    logging.warning(f"Session has been saved to '/sessions/{estimator_name}/{str(iso_ts)}.json'")


def load_latest_session(estimator_name: str):
    # include .pkl
    pkl_file = sorted([f for f in os.listdir(f"sessions/{estimator_name}/") if f.endswith(".pkl")])[-1]
    logging.warning(f"Session has been loaded from '/sessions/{estimator_name}/{pkl_file}'")
    return dill.load_session(f"sessions/{estimator_name}/{pkl_file}")  # Load the session
