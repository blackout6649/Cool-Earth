# Novel Mathematical Formulation presented in the paper

import numpy as np
from Deep_Space_Nav_VBN.src.utils.constants import SPEED_OF_LIGHT

def analytical_light_time_delay(r_pl_sc, v_pl):
    """
        Computes the light-time delay using the analytical approximation.

        Based on Eq. 37 in Andreis et al. (2024) .

        Args:
            r_pl_sc (np.array): Relative position vector (Planet - Spacecraft) [km]
            v_pl (np.array): Velocity vector of the planet [km/s]

        Returns:
            float: Light-time delay (delta_t) in seconds
    """
    c = SPEED_OF_LIGHT

    norm_r_pl_sc = np.linalg.norm(r_pl_sc)
    norm_v_pl_sc = np.linalg.norm(v_pl)
    norm_beta_pl = np.linalg.norm(v_pl) / SPEED_OF_LIGHT
    beta_pl_sqr = v_pl @ v_pl / c / c
    cos_epsilon = np.dot(r_pl_sc, v_pl) / (norm_r_pl_sc * norm_v_pl_sc)

    term1 = norm_r_pl_sc / c / (1 - beta_pl_sqr)
    term2 = -norm_beta_pl * cos_epsilon
    term3 = np.sqrt(beta_pl_sqr * (cos_epsilon**2 - 1) + 1)

    delta_t = term1 * (term2 + term3)

    return delta_t

def stellar_aberration_correction(los_true, v_sc):
    """
        Applies first-order relativistic stellar aberration to a Line of Sight vector.

        Based on Eq. 39 in Andreis et al. (2024)

        Args:
            los_true (np.array): The geometric (true) unit vector to the object.
            v_sc (np.array): The spacecraft's velocity vector [km/s].

        Returns:
            np.array: The aberrated (apparent) unit vector.
    """
    c = SPEED_OF_LIGHT
    beta_sc = v_sc / c

    cross_beta_l = np.cross(beta_sc, los_true)
    pertubation = np.cross(los_true, cross_beta_l)

    los_aberrated = los_true + pertubation

    # Re-normalize
    return los_aberrated / np.linalg.norm(los_true)