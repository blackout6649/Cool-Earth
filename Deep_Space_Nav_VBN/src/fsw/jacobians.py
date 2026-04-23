import numpy as np
from Deep_Space_Nav_VBN.src.utils.math_ops import skew_symmetric


def compute_R_sc(beta_sc, I_pl_sc, r_pl_tmDt, r_sc_t, r_pl_sc_t, v_pl_t, c):
    """
    Equation 57
    """
    # 1. Reshape inputs to column vectors (3,1)
    beta_sc = beta_sc.reshape(3, 1)
    I_pl_sc = I_pl_sc.reshape(3, 1)
    r_pl_tmDt = r_pl_tmDt.reshape(3, 1)
    r_sc_t = r_sc_t.reshape(3, 1)
    r_pl_sc_t = r_pl_sc_t.reshape(3, 1)
    v_pl_t = v_pl_t.reshape(3, 1)

    I3 = np.eye(3)

    # --- Term 1: Aberration Rotation ---
    # (I + 2 beta I^T - I^T beta I - I beta^T)
    term1 = (
            I3
            + 2 * beta_sc @ I_pl_sc.T
            - (I_pl_sc.T @ beta_sc) * I3
            - I_pl_sc @ beta_sc.T
    )

    # --- Term 2: Geometric Sensitivity ---
    # ( I / ||r|| - r r^T / ||r||^3 )
    delta_r = r_pl_tmDt - r_sc_t
    dist = np.linalg.norm(delta_r)

    # Safety check for division by zero
    if dist < 1e-6: dist = 1e-6

    term2 = (
            I3 / dist
            - (delta_r @ delta_r.T) / (dist ** 3)
    )

    # --- Term 3: Light-Time Sensitivity ---
    # Calculate scalar D = c^2 - v^T v
    v_norm_sq = (v_pl_t.T @ v_pl_t).item()
    D = c ** 2 - v_norm_sq

    # Big Fraction Components
    r_dot_r = (r_pl_sc_t.T @ r_pl_sc_t).item()
    r_dot_v = (r_pl_sc_t.T @ v_pl_t).item()

    # Numerator: (c^2 - v^2) r^T + (r^T v) v v^T
    numerator = (D * r_pl_sc_t.T) + (r_dot_v * v_pl_t @ v_pl_t.T)

    # Denominator: sqrt( (r^T r)(c^2 - v^2) + (r^T v)^2 )
    denom = np.sqrt(r_dot_r * D + r_dot_v ** 2)

    fraction_term = numerator / denom

    # USER SPECIFIC: ( v / D ) * ( Fraction - v^T ) - I
    term3 = (v_pl_t / D) @ (fraction_term - v_pl_t.T) - I3

    return term1 @ term2 @ term3


def compute_V_sc(I_pl_sc, c):
    """
    Equation 58
    """
    I_pl_sc = I_pl_sc.reshape(3, 1)
    I3 = np.eye(3)

    # Note: I^T I is the scalar dot product (usually 1.0 for unit vector)
    scalar_part = (I_pl_sc.T @ I_pl_sc).item()

    term = scalar_part * I3 - (I_pl_sc @ I_pl_sc.T)

    return (1.0 / c) * term


def compute_Q_att(I_pl_sc_aberr):
    """
    Equation 59
    """
    return 2.0 * skew_symmetric(I_pl_sc_aberr)


def compute_R_pl(beta_sc, I_pl_sc, r_pl_tmDt, r_sc_t, r_pl_sc_t, v_pl_t, c):
    """
    Equation 60
    """
    # Reshaping
    beta_sc = beta_sc.reshape(3, 1)
    I_pl_sc = I_pl_sc.reshape(3, 1)
    r_pl_tmDt = r_pl_tmDt.reshape(3, 1)
    r_sc_t = r_sc_t.reshape(3, 1)
    r_pl_sc_t = r_pl_sc_t.reshape(3, 1)
    v_pl_t = v_pl_t.reshape(3, 1)

    I3 = np.eye(3)

    # Term 1 & 2 (Same as R_sc)
    term1 = (I3 + 2 * beta_sc @ I_pl_sc.T - (I_pl_sc.T @ beta_sc) * I3 - I_pl_sc @ beta_sc.T)

    delta_r = r_pl_tmDt - r_sc_t
    dist = np.linalg.norm(delta_r)
    if dist < 1e-6: dist = 1e-6

    term2 = (I3 / dist - (delta_r @ delta_r.T) / (dist ** 3))

    # Term 3
    v_norm_sq = (v_pl_t.T @ v_pl_t).item()
    D = c ** 2 - v_norm_sq

    r_dot_r = (r_pl_sc_t.T @ r_pl_sc_t).item()
    r_dot_v = (r_pl_sc_t.T @ v_pl_t).item()

    numerator = (D * r_pl_sc_t.T) + (r_dot_v * v_pl_t @ v_pl_t.T)
    denom = np.sqrt(r_dot_r * D + r_dot_v ** 2)
    fraction_term = numerator / denom

    # USER SPECIFIC: I - (v / D) * ( Fraction - v^T )
    term3 = I3 - (v_pl_t / D) @ (fraction_term - v_pl_t.T)

    return term1 @ term2 @ term3


def compute_V_pl(beta_sc, I_pl_sc, r_pl_tmDt, r_sc_t, r_pl_sc_t, v_pl_t, c, dt):
    """
    Equation 61
    """
    # Reshaping
    beta_sc = beta_sc.reshape(3, 1)
    I_pl_sc = I_pl_sc.reshape(3, 1)
    r_pl_tmDt = r_pl_tmDt.reshape(3, 1)
    r_sc_t = r_sc_t.reshape(3, 1)
    r_pl_sc_t = r_pl_sc_t.reshape(3, 1)
    v_pl_t = v_pl_t.reshape(3, 1)
    I3 = np.eye(3)

    # Term 1 & 2 (Same as R_sc/R_pl)
    term1 = (I3 + 2 * beta_sc @ I_pl_sc.T - (I_pl_sc.T @ beta_sc) * I3 - I_pl_sc @ beta_sc.T)

    delta_r = r_pl_tmDt - r_sc_t
    dist = np.linalg.norm(delta_r)
    if dist < 1e-6: dist = 1e-6
    term2 = (I3 / dist - (delta_r @ delta_r.T) / (dist ** 3))

    # Term 3
    v_norm_sq = (v_pl_t.T @ v_pl_t).item()
    D = c ** 2 - v_norm_sq

    r_dot_r = (r_pl_sc_t.T @ r_pl_sc_t).item()
    r_dot_v = (r_pl_sc_t.T @ v_pl_t).item()

    # USER SPECIFIC: Numerator in V_pl is (r.v)r^T - (r.r)v^T
    numerator = r_dot_v * r_pl_sc_t.T - r_dot_r * v_pl_t.T
    denom = np.sqrt(r_dot_r * D + r_dot_v ** 2)
    fraction_term = numerator / denom

    # USER SPECIFIC: Tail term is (Fraction + 2v^T/D - r^T)
    tail_term = fraction_term + (2 * v_pl_t.T / D) - r_pl_sc_t.T

    # Full Term 3: Delta_t * I + (v / D) * Tail
    term3 = (dt * I3) + (v_pl_t / D) @ tail_term

    # Result is Negative
    return -1.0 * term1 @ term2 @ term3