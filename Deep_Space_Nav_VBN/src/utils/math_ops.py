# Helper functions
import numpy as np


def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3-element vector v.
    Implements the [v]^wedge notation from Eq. 215[cite: 215].

    Args:
        v (np.array): Shape (3,) or (3,1)

    Returns:
        np.array: 3x3 Skew-symmetric matrix
    """
    v = v.flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def normalize(v):
    """Safe normalization of a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm