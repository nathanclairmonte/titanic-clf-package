import math
import numpy as np

from classification_model.predict import make_prediction

def test_make_prediction(sample_input_data):
    # given
    # expected_num_preds = 0
    # expected_first_pred_value = 0

    # when
    result = make_prediction(input_data=sample_input_data)

    # then
    predictions = result.get("predictions") # using .get here in case key doesn't exist
    print(len(predictions))
    assert isinstance(predictions, np.ndarray)
    assert all(isinstance(x, np.int64) for x in predictions)
    assert result.get("errors") is None
    # assert len(predictions) == expected_num_preds
    # assert math.isclose(predictions[0], expected_first_pred_value, abs_tol=100)