import numpy as np
import pandas as pd
from wde import wde

def test_wde_numpy():
    # Create some dummy data
    x = np.random.randn(100)

    # Compute the wavelet density estimation
    hist, x_grid = wde(x)

    # Check that the output is valid
    assert isinstance(hist, np.ndarray)
    assert isinstance(x_grid, np.ndarray)
    assert len(hist) == len(x_grid)
    assert np.all(hist >= 0)

def test_wde_pandas():
    # Create some dummy data
    x = pd.Series(np.random.randn(100))

    # Compute the wavelet density estimation
    hist, x_grid = wde(x)

    # Check that the output is valid
    assert isinstance(hist, np.ndarray)
    assert isinstance(x_grid, np.ndarray)
    assert len(hist) == len(x_grid)
    assert np.all(hist >= 0)
