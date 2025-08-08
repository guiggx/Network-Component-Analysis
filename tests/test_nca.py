import numpy as np
from nca import NCA

def test_nca():
    # Create a dummy Z matrix (connectivity matrix)
    Z = np.zeros((5, 3))
    Z[0, 0] = 1
    Z[1, 0] = 1
    Z[2, 1] = 1
    Z[3, 1] = 1
    Z[4, 2] = 1

    # Create a dummy P matrix (transcription factor activities)
    P = np.random.rand(3, 10)

    # Create some dummy expression data
    E = np.random.rand(5, 10)

    # Create an NCA object
    nca = NCA(Z=Z, P=P)

    # Train the model
    nca.train(E, num_it=2)

    # Check that the results have the correct shape
    assert nca.A.shape == (5, 3)
    assert nca.P.shape == (3, 10)
