#################### IMPORTS ####################
#################################################


import numpy as np


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


class CostModel:
    def __init__(self, config):
        self.config = config
        self.U_SHAPE_ARR = np.ones((self.config.K, self.config.U_SPACE), dtype=self.config.DTYPE)

    def compute_state_cost(self, state, state_des):
        """
        Compute state-dependent cost
        """
        state_cost = (state - state_des) @ self.config.Q @ (state - state_des).T
        return np.diag(state_cost)

    def compute_control_cost(self, u, v):
        """
        Compute control-dependent cost
        """
        self.U_SHAPE_ARR[:] = u
        control_cost = self.U_SHAPE_ARR @ self.config.U_SIGMA_ARR @ v.T
        return self.config.GAMMA * np.diag(control_cost)

    def __call__(self, state, desired_state, u, v):
        total_cost = self.compute_state_cost(state, desired_state) + self.compute_control_cost(u, v)
        return total_cost