# Extended Kalman filter logic
import numpy as np
class NavigationFilter:
    def __init__(self, initial_guess):
        self.estimated_state = initial_guess
        self.P = np.eye(6) # Covariance matrix

    def predict(self, dt):
        # 1. Propagate state estimate (F matrix)
        # 2. Propagate covariance (P = FPF' + Q)
        return

    def update(self, measurement):
        # 1. Calculate Expected Measurement (h(x)) using Models.py
        # 2. Calculate Jacobian (H)
        # 3. Compute Kalman Gain (K)
        # 4. Update State and Covariance
        return
