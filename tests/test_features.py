from classification_model.config.core import config
from classification_model.processing.feature_transformers import ExtractLetterTransformer

def test_extract_letter_transformer(sample_input_data):
    # given
    assert sample_input_data[config.ml_config.cabin_var].iat[0] == "B5"
    assert sample_input_data[config.ml_config.cabin_var].iat[114] == "C23"
    assert sample_input_data[config.ml_config.cabin_var].iat[172] == "E60"
    transformer = ExtractLetterTransformer(variables=[config.ml_config.cabin_var])

    # when
    subject = transformer.fit_transform(sample_input_data)

    # then
    assert subject[config.ml_config.cabin_var].iat[0] == "B"
    assert subject[config.ml_config.cabin_var].iat[114] == "C"
    assert subject[config.ml_config.cabin_var].iat[172] == "E"