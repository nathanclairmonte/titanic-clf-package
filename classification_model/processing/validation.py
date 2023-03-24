from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config
from classification_model.processing.data_manager import _early_processing

# def drop_nan_rows(*, input_data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Function to drop NaN values from data that will be passed to
#     a trained model/pipeline.

#     We only drop NaN values from columns that are not going to be imputed
#     by our pipeline.

#     NB: Since in THIS case our pipeline imputes ALL missing values, this function
#     is left empty here.
#     """
#     pass

def validate_data(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Function to validate data for new data that will be passed
    to a trained model/pipeline.

    First performs the same early processing steps as in load_dataset(), then
    validates data with pydantic classes.
    """

    # validated_data = _early_processing(input_data)
    validated_data = input_data.copy()
    errors = None

    try:
        # replacing numpy nans before passing to pydantic, otherwise will break
        # NB not doing it in place, so original numpy nans will remain when returning df
        TitanicDataset(
            data=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

class TitanicDataRowSchema(BaseModel):
    survived: Optional[int]
    age: Optional[float]
    fare: Optional[float]
    sex: Optional[str]
    cabin: Optional[str]
    embarked: Optional[str]
    title: Optional[str]

class TitanicDataset(BaseModel):
    data: List[TitanicDataRowSchema]