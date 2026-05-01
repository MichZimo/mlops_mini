import numpy as np
from src.evaluate import adjusted_r2, mse
import pytest
import pandas as pd

@pytest.fixture
def data():
    return pd.Series([1, 2, 3])

def adjusted_r2_numerical(X, y_true, y_pred):
        y_true = y_true.to_numpy()
        y_pred = y_pred.to_numpy()
        mean_data = np.mean(y_true)
        n = len(y_true) # Number of datapoints
        p = X.shape[1] # Number of params
        ss_res = np.sum(np.square(y_true - y_pred))
        ss_tot = np.sum(np.square(y_true - mean_data))
        r2_score = 1 - ss_res / ss_tot
        return 1 -  (1 - r2_score) * (n - 1) / (n - p - 1)

## Test model evaluations
def test_mse_perfect(data):
    expected = np.mean((data - data) ** 2)
    assert isinstance(mse(data, data), float)
    assert mse(data, data) == pytest.approx(expected) # Float comparisons!

def test_adjusted_r2_perfect(data):
    X = pd.DataFrame([[1], [2], [3]])
    r2_perfect = adjusted_r2(X, data, data)
    assert isinstance(r2_perfect, float)
    assert r2_perfect == pytest.approx(adjusted_r2_numerical(X, data, data), rel=1e-3)

def test_adjusted_r2_worse_than_baseline(data):
    X = pd.DataFrame([[1], [2], [3]])
    y_pred = pd.Series([10, 10, 10])
    score = adjusted_r2(X, data, y_pred)
    assert score < 0

def test_shape_mismatch():
    y = pd.Series([0, 1])
    with pytest.raises(ValueError):
        mse(y, data)
