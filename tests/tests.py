import numpy as np
from src.evaluate import adjusted_r2, mse
import pytest
import pandas as pd

@pytest.fixture
def data():
    return pd.Series([1, 2, 3])

## Test model evaluations
def test_mse_perfect(data):
    expected = np.mean((data - data) ** 2)
    assert isinstance(mse_0, float)
    assert mse(data, data) == pytest.approx(expected) # Float comparisons!

def test_adjusted_r2_perfect(data):
    X = pd.Series([[1], [2], [3]])
    r2_perfect = adjusted_r2(X, data, data)
    assert isinstance(r2_perfect, float)
    assert r2_perfect == pytest.approx(1.0, rel=1e-3)

def test_adjusted_r2_worse_than_baseline(data):
    X = pd.Series([[1], [2], [3]])
    y_pred = pd.Series([10, 10, 10])
    score = adjusted_r2(X, data, y_pred)
    assert score < 0

def test_shape_mismatch():
    y = pd.Series([0, 1])
    mse_mismatch = mse(y, data)
    with pytest.raises(ValueError):
        mse(y, y_pred)

