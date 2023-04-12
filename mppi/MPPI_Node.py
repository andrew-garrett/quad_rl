import numpy as np
import cupy as cp





#################### MPPI PARAMETERS ####################
#########################################################

HOVER_RPM = 14468.429 # ================= RPM @ HOVER, NOMINAL CONTROL

X_SPACE = 12 # ========================== STATE SPACE (x,y,z, r,p,y, v_x,v_y,v_z, w_x,w_y,w_z)
U_SPACE = 4 # =========================== CONTROL SPACE
U_SIGMA = 10. # ========================= CONTROL NOISE

K = 512 # =============================== NUMBER OF TRAJECTORIES TO SAMPLE
T_HORIZON = 2.5 # ======================= TIME HORIZON
FREQUENCY = 48 # ======================== CONTROL FREQUENCY
T = int(T_HORIZON*FREQUENCY) # ======= NUMBER OF TIMESTEPS

TEMPERATURE = 1. # ====================== TEMPERATURE
GAMMA = 1. # ============================ CONTROL COST PARAMETER
ALPHA = 0.1 # =========================== NOMINAL CONTROL CENTERING PARAMETER


#################### FUNCTIONAL DYNAMICS MODEL ####################
###################################################################

"""
Our functional dynamics model is defined as follows:

F: X_SPACE x U_SPACE -> X_SPACE

X: the 12 dimensional state of the quadrotor
U: the 4 dimensional control of the quadrotor (RPMs of each motor)

There are three stages to the dynamics model:

1. Preprocess state and control
    a. State:
        i. As described in "Learning Quadrotor Dynamics Using Neural Network for Flight Control", due to the singularities
        of euler angles, we instead pass the sin() and cos() of the euler angles to the dynamics model.
        ii. Additionally, we also remove the position from the state
    b. Control:
        i. It may be advantageous to preprocess the controls such that they are in the form (total thrust, body_moments)

2. Compute dynamics update
    a. The core dynamics model computes the linear and angular accelerations of the quadrotor
        i. With the above preprocessing, the input to the dynamics model is (sin(euler_angles), cos(euler_angles), linear velocity, angular velocity, control)

3. Postprocess state
    a. Since the dynamics model computes linear and angular accelerations, we must postprocess to obtain the state of the desired shape
        i. Using kinematics, compute (delta_x, delta_euler_angles, delta_linear_velocity, delta_angular_velocity)
        ii. Apply the delta to the initial state

"""


#################### FUNCTIONAL COST MODEL ####################
###############################################################

"""
Our functional cost model is defined as follows:

J: X_SPACE x U_SPACE x U_SPACE -> R

X: the 12 dimensional state of the quadrotor
U: the 4 dimensional control of the quadrotor (RPMs of each motor)

The cost model has a state-dependent cost and a control-dependent cost.

1. State-Dependent Cost:
    a. c(x) = (x - x_des)^T @ Q @ (x - x_des) + (10e6)*C
        i. Essentially, we determine the noisy deviation of the current state from the desired state.
        ii. Q is the state-cost covariance matrix, which is just the identity matrix
        iii. C is an indicator variable for if the current state is a collision with an obstacle

2. Control-Dependent Cost:
    a. d(u, v) = lambda*(u^T @ Sigma^-1 @ v)
        i. Lambda is the temperature used to weight the control term
        ii. Sigma is the control-cost covariance matrix, which is defined by some desired noise variance
        iii. u is the current optimal control, while v is the perturbed optimal control

"""


#################### MPPI ALGORITHM CLASS ####################
##############################################################


class MPPI:
    """
    Class to perform MPPI Algorithm
    """
    def __init__(self) -> None:
        #################### Set MPPI Parameters
        # Dynamics Model Parameters
        self.X_SPACE = X_SPACE
        self.U_SPACE = U_SPACE
        self.U_SIGMA = U_SIGMA
        self.U_SIGMA_ARR = self.U_SIGMA * np.eye(self.U_SPACE)
        self.U_NOMINAL = HOVER_RPM * np.ones(self.U_SPACE)
        self.F = ... # Functional Dynamics Model

        # Samping Parameters
        self.K = K
        self.T_HORIZON = T_HORIZON
        self.FREQUENCY = FREQUENCY
        self.T = int(self.T_HORIZON*self.FREQUENCY)
        self.SAMPLE_X = np.zeros((self.K, self.X_SPACE))
        
        # Cost Model Parameters
        self.TEMPERATURE = TEMPERATURE
        self.GAMMA = GAMMA
        self.ALPHA = ALPHA
        self.COST_MAP = np.zeros(self.K)
        self.S = ... # Functional Cost Model

        # Quality of Life Parameters
        self.U = self.U_NOMINAL


    def compute_weights(self):
        """
        Compute weights for importance sampling

        Returns:
            weights: np.array(K) - the importance sampling weights
        """
        rho = np.min(self.COST_MAP)
        min_normed_cost_map = self.COST_MAP - rho
        eta_tilde = np.sum(np.exp((-self.TEMPERATURE**-1)*(min_normed_cost_map)))
        weights = (eta_tilde**-1)*np.exp((-self.TEMPERATURE**-1)*min_normed_cost_map)
        return weights
    
    def smooth_controls(self, weights, du):
        """
        Use the Savitsky-Golay Filter to smooth controls

        Parameters:
            - weights: np.array(K) - the importance sampling weights
            - du: np.array(K, T, U_SPACE) - the control perturbations
        """
        # sav_gol_filter()
        pass

    def mppi_iter(self, state):
        """
        Perform an iteration of MPPI

        Parameters:
            - state: np.array(X_SPACE) - the current state of the quadrotor

        Returns:
            - next_control: np.array(U_SPACE) - the next optimal control to execute
        """

        # sample random control perturbations
        du = cp.random.normal(loc=0., scale=self.U_SIGMA, size=(self.K, self.T, self.U_SPACE), dtype=np.float32)
        
        # prepare to sample in parallel
        self.SAMPLE_X[:] = state

        # iteratively sample T-length trajectories, K times in parallel
        for t in range(1, self.T+1):
            # Get the current optimal control
            u_tm1 = self.U[t-1]
            # Perturb the current optimal control
            v_tm1 = u_tm1 + du[:, t-1, :]
            # Approximate the next state for the perturbed optimal control
            self.SAMPLE_X = F(self.SAMPLE_X, v_tm1.reshape((self.K, self.U_SPACE)))
            # Compute the cost of taking the perturbed optimal control
            self.COST_MAP += self.compute_cost(self.SAMPLE_X, u_tm1, v_tm1)

        # Compute the importance sampling weights
        weights = self.compute_weights()

        # Compute the smoothed control, weighted by importance sampling
        self.U += self.smooth_controls(weights, du)

        # Get the next control
        next_control = self.U[0]
        
        # Roll the controls
        self.U = np.roll(self.U, shift=-1, axis=0)
        self.U[-1] = self.U_NOMINAL

        return next_control
    


