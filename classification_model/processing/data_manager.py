import re
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def _get_first_cabin(cabin_entry):
    """
    Function to retain only the first cabin in a cabin_entry if there are more
    than one. If there are none, will give NaN. Will be applied to cabin column.
    """

    try:
        return cabin_entry.split()[0]
    except AttributeError:
        return np.nan


def _get_title(passenger):
    """
    Function to extract the title (Mr, Mrs, etc.) from the name column
    """

    if re.search("Mrs", passenger):
        return "Mrs"
    elif re.search("Mr", passenger):
        return "Mr"
    elif re.search("Miss", passenger):
        return "Miss"
    elif re.search("Master", passenger):
        return "Master"
    else:
        return "Other"


def _early_processing(data: pd.DataFrame) -> pd.DataFrame:
    """Performs early processing steps on data."""

    # replace question marks with NaN
    data = data.replace("?", np.nan)

    # retain only first cabin in each row
    data[config.ml_config.cabin_var] = data[config.ml_config.cabin_var].apply(
        _get_first_cabin
    )

    # retain only titles from the name variable
    data[config.ml_config.title_var] = data[config.ml_config.name_var].apply(_get_title)

    # cast numerical variables to floats
    for var in config.ml_config.num_vars:
        data[var] = data[var].astype("float")

    # drop unnecessary variables
    data.drop(labels=config.ml_config.drop_vars, axis=1, inplace=True)

    return data


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Function to load a given dataset into a pandas dataframe"""

    data = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))

    # early data processing steps
    data = _early_processing(data)

    return data


def load_pipeline(*, file_name: str) -> Pipeline:
    """Function to load a saved pipeline"""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Removes old saved pipelines.mThis is to make sure that there is only ever
    one saved pipeline.

    Want a simple one-to-one mapping between package version and model version
    that will be imported and used by other applications.

    Whenever we save a new pipeline, we will run this function to remove old
    pipelines (and keep the pipeline that was just saved).
    """

    files_to_keep += ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in files_to_keep:
            model_file.unlink()


def save_pipeline(*, pipeline_to_save: Pipeline) -> None:
    """
    Persists the pipeline.

    Saves the versioned model (pipeline), and overwrites any previously
    saved models. This ensures that when the package is published, there is only
    one trained model that can be called.
    """

    # prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_save, save_path)
