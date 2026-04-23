import numpy as np
from Deep_Space_Nav_VBN.src.fsw.models import analytical_light_time_delay, stellar_aberration_correction


class CameraModel:
    def __init__(self, K_matrix, resolution=(1024, 1024)):
        """
        Initialize the Camera Model.

        Args:
            K_matrix (np.array): 3x3 Intrinsic Matrix.
            resolution (tuple): (width, height) in pixels.
        """
        self.K = K_matrix
        self.width = resolution[0]
        self.height = resolution[1]

    def project_to_pixels(self, vector_camera_frame):
        """
        Projects a 3D vector in the camera frame onto the 2D image plane.
        Implements Eq. 40 and Eq. 41.

        Args:
            vector_camera_frame (np.array): 3x1 vector [x, y, z] in camera frame.

        Returns:
            np.array: 2x1 pixel coordinates [u, v].
        """

        # Homogeneous projection (Eq. 40 concept)
        # Multiply Intrinsic Matrix K with the 3D vector
        h_vec = self.K @ vector_camera_frame

        # Normalize by depth (Eq. 41)
        # u = x' / z'
        # v = y' / z'
        u = h_vec[0] / h_vec[2]
        v = h_vec[1] / h_vec[2]

        return np.array([u, v])

    def measurement_function(self, state_sc, state_pl, R_inertial_to_camera):
        """
        The full Measurement Model h(x).
        Predicts the pixel coordinates of a planet given the spacecraft state.

        Logic Flow:
        1. Calculate Light-Time Delay (Eq. 37)
        2. Update Planet Position to 'Emission Time'
        3. Calculate True Line-of-Sight (LoS)
        4. Apply Stellar Aberration (Eq. 39)
        5. Rotate to Camera Frame
        6. Project to Pixels (Eq. 41)

        Args:
            state_sc (np.array): Spacecraft State [rx, ry, rz, vx, vy, vz] (km, km/s)
            state_pl (np.array): Planet State [rx, ry, rz, vx, vy, vz] (km, km/s)
            R_inertial_to_camera (np.array): 3x3 Rotation Matrix (Attitude)

        Returns:
            np.array: Predicted measurement [u, v] in pixels.
        """
        # --- 1. Unpack States ---
        r_sc = state_sc[0:3]
        v_sc = state_sc[3:6]

        r_pl = state_pl[0:3]
        v_pl = state_pl[3:6]

        # --- 2. Initial Relative Position ---
        # Used only to estimate the time delay
        r_pl_sc_initial = r_pl - r_sc # r_{pl/sc}

        # --- 3. Light-Time Correction (The "Novel" Math) ---
        # Calculates delta_t based on Eq. 37
        dt = analytical_light_time_delay(r_pl_sc_initial, v_pl)

        # Correct the Planet Position
        # r_pl(t - dt) approx r_pl(t) - v_pl(t) * dt
        r_pl_delayed = r_pl - (v_pl * dt) # r_pl(t-dt)

        # --- 4. True Line of Sight (Inertial) ---
        # Re-calculate relative vector using the delayed planet position
        r_rel_delayed = r_pl_delayed - r_sc

        # Normalize to get the unit vector direction
        dist = np.linalg.norm(r_rel_delayed)

        los_inertial = r_rel_delayed / dist # Eq 38.

        # --- 5. Stellar Aberration Correction ---
        # Eq. 39: Warps the vector based on spacecraft velocity
        los_aberrated = stellar_aberration_correction(los_inertial, v_sc)

        # --- 6. Frame Rotation (Attitude) ---
        # Rotate from Inertial Frame to Camera Frame
        # v_camera = R * v_inertial -> Line of Sight vector in camera frame, corrected for time delay and aberration
        los_camera = R_inertial_to_camera @ los_aberrated # los^{aberr}_{pl/sc}

        # --- 7. Project to Pixels ---
        pixels = self.project_to_pixels(los_camera) # [x, y, z] -> [u, v]

        return pixels