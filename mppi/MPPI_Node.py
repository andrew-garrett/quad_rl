#################### IMPORTS ####################
#################################################


import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import json
from copy import deepcopy
from collections import namedtuple
import xml.etree.ElementTree as etxml
try:
    import cupy as cp
    """
        To migrate to cupy over numpy, we need the following commands from cupy:
            np.array() <-> cp.array()
            np.sum(x, axis, dtype) <-> cp.sum(x, axis, dtype)
    """
except:
    print("cupy not available, defaulting to np/torch")
import numpy as np
from scipy.signal import savgol_filter
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.random
import mppi.dynamics_models as dynamics_models
import mppi.cost_models as cost_models


#################### MPPI PARAMETERS ####################
#########################################################


def get_mppi_config(config_fpath="./configs/mppi_config.json"):
    """
    Creates a config object for mppi parameters

    Parameters:
        - config_fpath: str - the filepath of the desired config file

    Returns:
        - config: namedtuple() - the mppi config object
    """
    with open(config_fpath, "r") as f:
        config_dict = json.load(f)

    # Get URDF parameters
    config_dict["CF2X"] = parseURDFParameters(os.path.join(os.path.dirname(config_fpath), "cf2x.urdf"))
    # Set derived/processed parameters
    config_dict["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    config_dict["T"] = int(config_dict["T_HORIZON"] * config_dict["FREQUENCY"])
    config_dict["METHOD"] = config_dict["METHOD"].lower()
    if config_dict["METHOD"] in ("torch", "cupy"):
        config_dict["DYNAMICS_MODEL"] = config_dict["METHOD"].capitalize() + config_dict["DYNAMICS_MODEL"]
        if "cupy" in config_dict["METHOD"]:
            config_dict["METHOD"] = cp
            config_dict["DTYPE"] = getattr(np, config_dict["DTYPE"])
        else:
            config_dict["METHOD"] = torch
            config_dict["DTYPE"] = getattr(config_dict["METHOD"], config_dict["DTYPE"])
    else: 
        config_dict["METHOD"] = np
        config_dict["DTYPE"] = getattr(config_dict["METHOD"], config_dict["DTYPE"])

    # For wandb sweeps
    if "Q" not in config_dict.keys():
        Q = np.ones(config_dict["X_SPACE"])
        if "Q_p" in config_dict.keys():
            Q[:3] = np.array([config_dict["Q_p"] for i in range(3)])
        if "Q_r" in config_dict.keys():
            Q[3:6] = np.array([config_dict["Q_r"] for i in range(3)])
        if "Q_v" in config_dict.keys():
            Q[6:9] = np.array([config_dict["Q_v"] for i in range(3)])
        if "Q_w" in config_dict.keys():
            Q[9:12] = np.array([config_dict["Q_w"] for i in range(3)])
        config_dict["Q"] = Q

    config_dict["Q"] = np.diag(config_dict["Q"])
    
    config_dict["U_NOMINAL"] = config_dict["CF2X"].HOVER_RPM * np.ones(config_dict["U_SPACE"])
    config_dict["SYSTEM_BIAS"] = config_dict["SYSTEM_BIAS"] * np.ones(config_dict["U_SPACE"])
    config_dict["SYSTEM_NOISE"] = config_dict["SYSTEM_NOISE"] * np.eye(config_dict["U_SPACE"])
    config_dict["DT"] = 1.0/config_dict["FREQUENCY"]
    config_dict["DISCOUNT"] = 1.0 - config_dict["DT"]

    config = namedtuple("mppi_config", config_dict.keys())(**config_dict)
    return config

def parseURDFParameters(urdf_fpath="./configs/cf2x.urdf"):
    """
    Loads parameters from the URDF file.

    This is almost identical to the one in the BaseAviary class

    Parameters:
        - urdf_fpath: str - the filepath of the desired urdf config file

    Returns:
        - config: namedtuple() - the drone_model urdf config object
    """
    URDF_TREE = etxml.parse(urdf_fpath).getroot()
    M = float(URDF_TREE[1][0][1].attrib['value'])
    L = float(URDF_TREE[0].attrib['arm'])
    THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
    IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
    IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
    IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
    J = np.diag([IXX, IYY, IZZ])
    J_INV = np.linalg.inv(J)
    KF = float(URDF_TREE[0].attrib['kf'])
    KM = float(URDF_TREE[0].attrib['km'])
    COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
    COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
    COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
    COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
    MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
    GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
    PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
    DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
    DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
    DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
    DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
    DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
    DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
    
    # Include MAX_RPM and HOVER_RPM
    G = 9.8
    GRAVITY = G*M
    HOVER_RPM = np.sqrt(GRAVITY / (4*KF))
    MAX_RPM = np.sqrt((THRUST2WEIGHT_RATIO*GRAVITY) / (4*KF))
    # Put these parameters into a dictionary
    config_dict = locals()
    # Remove unneccesary values
    config_dict.pop("URDF_TREE")
    config_dict.pop("urdf_fpath")
    # Return a namedtuple
    config = namedtuple("cf2x_config", config_dict.keys())(**config_dict)
    return config


"""
MPPI_CONFIG PARAMETERS AND DEFAULT VALUES:

HOVER_RPM = 14468.429 # ============================================== RPM @ HOVER, NOMINAL CONTROL
U_NOMINAL = [HOVER_RPM, HOVER_RPM, HOVER_RPM, HOVER_RPM] # =========== NOMINAL CONTROL (HOVERING)

X_SPACE = 12 # ======================================================= STATE SPACE (x,y,z, r,p,y, v_x,v_y,v_z, w_x,w_y,w_z)
U_SPACE = 4 # ======================================================== CONTROL SPACE
U_SIGMA = 10. # ====================================================== CONTROL NOISE
SYSTEM_NOISE = np.linalg.inv(U_SIGMA * np.eye(U_SPACE)) ============== CONTROL-COST COVARIANCE

K = 512 # ============================================================ NUMBER OF TRAJECTORIES TO SAMPLE
T_HORIZON = 2.5 # ==================================================== TIME HORIZON
FREQUENCY = 48 # ===================================================== CONTROL FREQUENCY
DT = 1.0 / FREQUENCY # =============================================== TIMESTEP SIZE
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
    def __init__(self, config, state_des) -> None:

        # Set MPPI Parameters
        self.config = config
        self.METHOD = self.config.METHOD

        # Functional Dynamics Model
        try:
            f_config = self.config
            self.F = getattr(dynamics_models, f_config.DYNAMICS_MODEL)(f_config)
        except:
            self.F = dynamics_models.DynamicsModel(f_config)
        
        # Functional Cost Model
        try:
            s_config = self.config
            self.S = getattr(cost_models, self.config.COST_MODEL)(s_config, state_des=state_des)
        except:
            self.S = cost_models.CostModel(s_config, state_des=state_des)

        # Pre-allocate data structures
        self.COST_MAP = self.METHOD.zeros(self.config.K, dtype=self.config.DTYPE)
        self.SAMPLES_X = self.METHOD.zeros((self.config.T+1, self.config.K, self.config.X_SPACE), dtype=self.config.DTYPE)
        self.U = self.METHOD.zeros((self.config.T, self.config.U_SPACE), dtype=self.config.DTYPE)
        U_NOMINAL = self.METHOD.asarray(self.config.U_NOMINAL, dtype=self.config.DTYPE)
        if self.METHOD.__name__ in ("numpy", "cupy"):
            self.U_NOMINAL = U_NOMINAL.copy()
            self.SYSTEM_BIAS = self.METHOD.asarray(self.config.SYSTEM_BIAS, dtype=self.config.DTYPE)
            self.mu_sigma = (self.SYSTEM_BIAS, self.METHOD.asarray(self.config.SYSTEM_NOISE, dtype=self.config.DTYPE))
            self.noise_dist = lambda size: self.METHOD.random.multivariate_normal(self.mu_sigma[0], self.mu_sigma[1], size)
        else:
            self.U_NOMINAL = U_NOMINAL.clone().to(device=self.config.DEVICE)
            self.SYSTEM_BIAS = self.METHOD.asarray(self.config.SYSTEM_BIAS).to(device=self.config.DEVICE, dtype=self.config.DTYPE)
            self.mu_sigma = (self.SYSTEM_BIAS, self.METHOD.asarray(self.config.SYSTEM_NOISE).to(device=self.config.DEVICE, dtype=self.config.DTYPE))
            self.COST_MAP, self.SAMPLES_X, self.U = self.COST_MAP.to(self.config.DEVICE), self.SAMPLES_X.to(self.config.DEVICE), self.U.to(self.config.DEVICE)
            self.noise_dist_obj = torch.distributions.MultivariateNormal(loc=self.mu_sigma[0], covariance_matrix=self.mu_sigma[1])
            self.noise_dist = lambda size: self.noise_dist_obj.rsample(size)
        self.U = self.noise_dist(self.config.T) + self.U_NOMINAL

    def reset(self, desired_state=None):
        """
        Resets the controller, unless an argument is given, where it only changes the set-point.
        """
        if desired_state is not None:
            self.S.set_new_desired_state(desired_state)
            return
        self.SAMPLES_X = self.METHOD.zeros((self.config.T+1, self.config.K, self.config.X_SPACE), dtype=self.config.DTYPE)
        self.U = self.noise_dist(self.config.T) + self.U_NOMINAL

    def compute_weights(self):
        """
        Compute weights for importance sampling

        Returns:
            weights: np.array(K) - the importance sampling weights
        """
        rho = self.METHOD.min(self.COST_MAP)
        min_normed_cost_map = self.COST_MAP - rho
        weights = self.METHOD.exp(-(self.config.TEMPERATURE**-1)*min_normed_cost_map)
        self.weights = weights / self.METHOD.sum(weights)
    
    def smooth_controls(self, du):
        """
        Use the Savitsky-Golay Filter to smooth controls

        Parameters:
            - weights: np.array(K) - the importance sampling weights
            - du: np.array(K, T, U_SPACE) - the control perturbations
        """
        weighted_samples = du.T @ self.weights
        # self.U += weighted_samples.T
        self.U += savgol_filter(
            weighted_samples.T, 
            window_length=int(0.1*self.config.T), 
            polyorder=5, 
            axis=0
        )

    def command(self, state, shift_nominal_trajectory=True):
        """
        Perform an iteration of MPPI

        Parameters:
            - state: np.array(X_SPACE) - the current state of the quadrotor

        Returns:
            - next_control: np.array(U_SPACE) - the next optimal control to execute
        """
        # Reset for the current iteration
        self.COST_MAP = self.METHOD.zeros_like(self.COST_MAP)

        # sample random control perturbations
        du = self.noise_dist((self.config.K, self.config.T))
        if self.METHOD.__name__ in ("cupy", "numpy"):
            self.SAMPLES_X[0, :] = self.METHOD.array(state)
        else:
            self.SAMPLES_X[0, :] = state.clone()

        # iteratively sample T-length trajectories, K times in parallel
        curr_discount = 1.0
        for t in range(1, self.config.T+1):
            curr_discount *= self.config.DISCOUNT
            # Get the current control
            u_tm1 = self.U[t-1]
            # Perturb the current control
            du_tm1 = du[:, t-1, :]
            v_tm1 = u_tm1 + du_tm1
            # Approximate the next state for the perturbed current control
            self.SAMPLES_X[t] = self.F(self.SAMPLES_X[t-1], v_tm1)
            # Compute the cost of taking the perturbed optimal control (DISCOUNTED BY HOW FAR THROUGH THE TRAJECTORY WE ARE)
            self.COST_MAP += curr_discount*self.S(self.SAMPLES_X[t], (u_tm1, du_tm1))

        # Compute the importance sampling weights
        self.compute_weights()

        # Compute the smoothed control, weighted by importance sampling
        self.smooth_controls(du)

        # Get the next control
        next_control = self.U[0]
        if shift_nominal_trajectory:
            # Roll the controls
            self.U = self.METHOD.roll(self.U, -1, 0)
            self.U[-1] = self.U_NOMINAL
        
        return next_control
    


if __name__ == "__main__":

    mppi_config = get_mppi_config()
    mppi_node = MPPI(mppi_config)

    # test an iteration of MPPI

    x0 = np.zeros(mppi_config.X_SPACE, dtype=mppi_config.DTYPE)
    x0[2] = 1.
    xdes = np.zeros_like(x0, dtype=mppi_config.DTYPE)
    xdes[2] = 2.

    next_control = mppi_node.command(x0, xdes)
    print(next_control)