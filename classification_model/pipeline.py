from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from feature_engine.imputation import CategoricalImputer, AddMissingIndicator, MeanMedianImputer
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder

from classification_model.config.core import config
from classification_model.processing import feature_transformers as ft

pipeline = Pipeline([
    # ------ IMPUTATION ------

    # impute categorical vars with the string "Missing"
    ('categorical_imputation', CategoricalImputer(imputation_method='missing',
                                                  variables=config.ml_config.cat_vars)),

    # add missing indicator for numerical variables
    ('missing_indicator', AddMissingIndicator(variables=config.ml_config.num_vars)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(imputation_method='median',
                                            variables=config.ml_config.num_vars)),

    # extract letter from each cabin entry
    ('letter_extractor', ft.ExtractLetterTransformer(variables=config.ml_config.cabin_var)),

    
    # ------ CATEGORICAL ENCODING ------

    # remove categories present in less than 5% of the observations
    # group them into one category called "Rare"
    ('rare_label_encoder', RareLabelEncoder(tol=0.05,
                                            n_categories=1,
                                            variables=config.ml_config.cat_vars)),

    # encode categorical variables using one-hot encoding into k-1 vars (i.e. drop last)
    ('categorical_encoder', OneHotEncoder(drop_last=True,
                                          variables=config.ml_config.cat_vars)),


    # ------ SCALING ------
    ('scaler', StandardScaler()),

    # ------ CLASSIFIER ------
    ('clf', LogisticRegression(C=config.ml_config.log_reg_c,
                               random_state=config.ml_config.random_state))
])