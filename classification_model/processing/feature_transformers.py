from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    """
    Extract the first letter of the cabin variable.

    i.e. E22 -> E
    """

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("The variables parameter must be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # just including this function to match sklearn pipeline
        return self

    def transform(self, X):

        # copying so we don't overwrite original df
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[feature].str[0]

        return X
