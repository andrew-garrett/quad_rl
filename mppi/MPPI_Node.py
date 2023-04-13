#################### IMPORTS ####################
#################################################


import json
from collections import namedtuple
import numpy as np
try:
    import cupy as cp
except:
    print("cupy not available")
from scipy.signal import savgol_filter

import dynamics_models
import cost_models


#################### MPPI PARAMETERS ####################
#########################################################


def get_mppi_config(config_fpath="./mppi/configs/mppi_config.json"):
    """
    Creates a config object for more concise parameter storage

    Parameters:
        - config_fpath: str - the filepath of the desired config file

    Returns:
        - mppi_config: NamedTuple() - the config object
    """
    with open(config_fpath, "r") as f:
        config_dict = json.load(f)

        # Set derived/processed parameters
        config_dict["T"] = int(config_dict["T_HORIZON"] * config_dict["FREQUENCY"])
        config_dict["DTYPE"] = getattr(np, config_dict["DTYPE"])
        config_dict["Q"] = np.diag(config_dict["Q"])
        config_dict["U_NOMINAL"] = config_dict["HOVER_RPM"] * np.ones(config_dict["U_SPACE"], dtype=config_dict["DTYPE"])
        config_dict["U_SIGMA_ARR"] = np.linalg.inv(config_dict["U_SIGMA"] * np.eye(config_dict["U_SPACE"], dtype=config_dict["DTYPE"]))

    mppi_config = namedtuple("mppi_config", config_dict.keys())(**config_dict)
    return mppi_config


"""
HOVER_RPM = 14468.429 # ============================================== RPM @ HOVER, NOMINAL CONTROL
U_NOMINAL = [HOVER_RPM, HOVER_RPM, HOVER_RPM, HOVER_RPM] # =========== NOMINAL CONTROL (HOVERING)

X_SPACE = 12 # ======================================================= STATE SPACE (x,y,z, r,p,y, v_x,v_y,v_z, w_x,w_y,w_z)
U_SPACE = 4 # ======================================================== CONTROL SPACE
U_SIGMA = 10. # ====================================================== CONTROL NOISE
U_SIGMA_ARR = np.linalg.inv(U_SIGMA * np.eye(U_SPACE)) =============== CONTROL-COST COVARIANCE

K = 512 # ============================================================ NUMBER OF TRAJECTORIES TO SAMPLE
T_HORIZON = 2.5 # ==================================================== TIME HORIZON
FREQUENCY = 48 # ===================================================== CONTROL FREQUENCY
T = int(T_HORIZON*FREQUENCY) # ======================================= NUMBER OF TIMESTEPS

TEMPERATURE = 1. # =================================================== TEMPERATURE
GAMMA = 0.5 # ========================================================= CONTROL COST PARAMETER
ALPHA = 0.1 # ======================================================== NOMINAL CONTROL CENTERING PARAMETER
Q = np.eye(X_SPACE) # ================================================ STATE-COST COVARIANCE

DTYPE = np.float32 # ================================================= DATATYPE (np.float32 or np.float64)
"""


#################### MPPI ALGORITHM CLASS ####################
##############################################################


class MPPI:
    """
    Class to perform MPPI Algorithm
    """
    def __init__(self, config) -> None:

        # Set MPPI Parameters
        self.config = config

        try:
            self.F = getattr(dynamics_models, self.config.DYNAMICS_MODEL)(self.config) # Functional Dynamics Model
        except:
            self.F = dynamics_models.DynamicsModel(self.config)
        try:
            self.S = getattr(cost_models, self.config.COST_MODEL)(self.config) # Functional Cost Model
        except:
            self.S = cost_models.CostModel(self.config)

        # Pre-allocate data structures
        self.COST_MAP = np.zeros(self.config.K, dtype=self.config.DTYPE)
        self.SAMPLES_X = np.zeros((self.config.K, self.config.X_SPACE), dtype=self.config.DTYPE)
        self.U = np.zeros((self.config.T, self.config.U_SPACE), dtype=self.config.DTYPE)
        self.U[:] = self.config.U_NOMINAL

    def compute_weights(self):
        """
        Compute weights for importance sampling

        Returns:
            weights: np.array(K) - the importance sampling weights
        """
        rho = np.min(self.COST_MAP)
        min_normed_cost_map = self.COST_MAP - rho
        weights = np.exp(-(self.config.TEMPERATURE**-1)*min_normed_cost_map)
        return weights / np.sum(weights)
    
    def smooth_controls(self, weights, du):
        """
        Use the Savitsky-Golay Filter to smooth controls

        Parameters:
            - weights: np.array(K) - the importance sampling weights
            - du: np.array(K, T, U_SPACE) - the control perturbations
        """
        weighted_samples = du.T @ weights
        self.U += savgol_filter(
            weighted_samples.T, 
            window_length=int(np.sqrt(self.config.K)), 
            polyorder=7, 
            axis=0
        )

    def mppi_iter(self, state, desired_state):
        """
        Perform an iteration of MPPI

        Parameters:
            - state: np.array(X_SPACE) - the current state of the quadrotor

        Returns:
            - next_control: np.array(U_SPACE) - the next optimal control to execute
        """
        # sample random control perturbations
        try:
            du = cp.asnumpy(
                cp.random.normal(
                    loc=0., 
                    scale=self.config.U_SIGMA, 
                    size=(self.config.K, self.config.T, self.config.U_SPACE), 
                    dtype=self.config.DTYPE
                )
            )
        except:
            du = np.random.normal(
                loc=0., 
                scale=self.config.U_SIGMA, 
                size=(self.config.K, self.config.T, self.config.U_SPACE)
            )
        
        # prepare to sample in parallel
        self.SAMPLES_X[:] = state

        # iteratively sample T-length trajectories, K times in parallel
        for t in range(1, self.config.T+1):
            # Get the current optimal control
            u_tm1 = self.U[t-1]
            # Perturb the current optimal control
            v_tm1 = u_tm1 + du[:, t-1, :]
            # Approximate the next state for the perturbed optimal control
            self.SAMPLES_X = self.F(self.SAMPLES_X, v_tm1)
            # Compute the cost of taking the perturbed optimal control
            self.COST_MAP += self.S(self.SAMPLES_X, desired_state, u_tm1, v_tm1)

        # Compute the importance sampling weights
        weights = self.compute_weights()

        # Compute the smoothed control, weighted by importance sampling
        self.smooth_controls(weights, du)

        # Get the next control
        next_control = self.U[0]
        
        # Roll the controls
        self.U = np.roll(self.U, shift=-1, axis=0)
        self.U[-1] = self.config.U_NOMINAL

        return next_control
    


if __name__ == "__main__":

    mppi_config = get_mppi_config()
    mppi_node = MPPI(mppi_config)

    # test an iteration of MPPI

    x0 = np.zeros(mppi_config.X_SPACE, dtype=mppi_config.DTYPE)
    x0[2] = 1.
    xdes = np.zeros_like(x0, dtype=mppi_config.DTYPE)
    xdes[2] = 2.

    next_control = mppi_node.mppi_iter(x0, xdes)
    print(next_control)