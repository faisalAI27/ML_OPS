import pandas as pd
import numpy as np
import pytest

from src.validation.model_checks import check_model_predictions


class DummyRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_check_model_predictions_pass():
    model = DummyRegressor(3.5)
    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    check_model_predictions(model, X)  # should not raise


def test_check_model_predictions_out_of_range():
    model = DummyRegressor(6.0)
    X = pd.DataFrame({"f1": [1, 2]})
    with pytest.raises(ValueError):
        check_model_predictions(model, X)
