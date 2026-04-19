import numpy as np 
import pytest
from src.config import (SEQ_LENGTH, INPUT_SIZE, HORIZONS)
from src.predict import predict


def test_predict_output_shape_and_type():
    fake_input = np.random.randn(SEQ_LENGTH,INPUT_SIZE)
    for h in HORIZONS:
        result = predict(fake_input,horizon=h)
        assert isinstance(result, float), f"Expected float, got{type(result)}"

def test_predict_wrong_shape_raises():
    bad_input = np.random.randn(10, INPUT_SIZE)  # wrong sequence length
    with pytest.raises(ValueError):
        predict(bad_input, horizon=1)

def test_predict_wrong_horizon_raises():
    fake_input = np.random.randn(SEQ_LENGTH, INPUT_SIZE)
    with pytest.raises(ValueError):
        predict(fake_input, horizon=99)