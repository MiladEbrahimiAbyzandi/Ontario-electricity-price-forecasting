import numpy as np 
import pytest
from src.config import DATA_CFG
from src.predict import predict


def test_predict_output_shape_and_type():
    fake_input = np.random.randn(DATA_CFG.seq_length, DATA_CFG.input_size)
    for h in DATA_CFG.horizons:
        result = predict(fake_input, horizon=h)
        assert isinstance(result, float), f"Expected float, got{type(result)}"

def test_predict_wrong_shape_raises():
    bad_input = np.random.randn(10, DATA_CFG.input_size)  # wrong sequence length
    with pytest.raises(ValueError):
        predict(bad_input, horizon=1)

def test_predict_wrong_horizon_raises():
    fake_input = np.random.randn(DATA_CFG.seq_length, DATA_CFG.input_size)
    with pytest.raises(ValueError):
        predict(fake_input, horizon=99)