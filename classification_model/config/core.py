from pathlib import Path
from typing import List, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import classification_model

# project directories
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yaml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

# defining pydantic classes to validate config data
class AppConfig(BaseModel):
    """
    Application-level config validation
    """

    package_name: str
    train_data_file: str
    test_data_file: str
    pipeline_name: str
    pipeline_save_file: str

class MLConfig(BaseModel):
    """
    All config relevant to model training and feature engineering
    """

    target: str
    features: List[str]
    num_vars: List[str]
    cat_vars: List[str]
    drop_vars: List[str]
    cabin_var: str
    title_var: str
    name_var: str
    test_size: float
    log_reg_c: float
    random_state: float

class Config(BaseModel):
    """Master config object"""

    app_config: AppConfig
    ml_config: MLConfig

# helper functions
def find_config_file() -> Path:
    """Function to locate the config file"""

    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config file not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(config_path: Optional[Path] = None) -> YAML:
    """Parse YAML file containing package config info"""

    if not config_path:
        config_path = find_config_file()

    if config_path:
        with open(config_path, "r") as config_file:
            parsed_config = load(config_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {config_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Validate config values with pydantic classes"""

    if not parsed_config:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        ml_condfig=MLConfig(**parsed_config.data)
    )

    return _config

config = create_and_validate_config()