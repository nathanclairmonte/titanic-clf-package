import numpy as np
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.pipeline import pipeline
from classification_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    """Function to train the model."""

    # load train data
    data = load_dataset(file_name=config.app_config.train_data_file)

    # divide into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.ml_config.features],
        data[config.ml_config.target],
        test_size=config.ml_config.test_size,
        random_state=config.ml_config.random_state
    )

    # fit model/pipeline
    pipeline.fit(X_train, y_train)

    # save trained model/pipeline
    save_pipeline(pipeline_to_save=pipeline)

if __name__=="__main__":
    run_training()