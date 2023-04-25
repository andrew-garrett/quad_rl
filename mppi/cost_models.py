#################### IMPORTS ####################
#################################################

try:
    import cupy as cp
except:
    print("Cupy not found, defaulting back to np/torch.")
import numpy as np
import torch


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
    """
    Basic Cost Model

    Parameterized by mppi_config, and uses a METHOD in (numpy, cupy, and torch) for array operations
    """
    def __init__(self, config, state_des):
        self.config = config
        self.METHOD = self.config.METHOD

        self.Q = self.METHOD.asarray(self.config.Q, dtype=self.config.DTYPE) # CP and NP and TORCH
        self.SYSTEM_NOISE_INV = np.linalg.inv(self.config.SYSTEM_NOISE) # CP and NP
        self.SYSTEM_NOISE_INV = self.METHOD.asarray(self.SYSTEM_NOISE_INV, dtype=self.config.DTYPE) # CP and NP and TORCH
        self.U_SHAPE_ARR = self.METHOD.ones((self.config.K, self.config.U_SPACE)) # CP and NP and TORCH
        self.U_SHAPE_ARR = self.METHOD.asarray(self.U_SHAPE_ARR, dtype=self.config.DTYPE) # CP and NP and TORCH
        if state_des is not None:
            self.state_des = self.METHOD.asarray(state_des, dtype=self.config.DTYPE) # CP and NP and TORCH
            if len(state_des.shape) <= 1:
                self.state_des = state_des.reshape(1, -1) # CP and NP and TORCH
            else:
                self.state_des = state_des
            if self.METHOD.__name__ == "torch":
                self.state_des = self.state_des.to(device=self.config.DEVICE)
        if self.METHOD.__name__ == "torch":
            self.Q = self.Q.to(device=self.config.DEVICE)
            self.SYSTEM_NOISE_INV = self.SYSTEM_NOISE_INV.to(device=self.config.DEVICE)
            self.U_SHAPE_ARR = self.U_SHAPE_ARR.to(device=self.config.DEVICE)

    def set_new_desired_state(self, state_des):
        self.state_des = self.METHOD.asarray(state_des, dtype=self.config.DTYPE) # CP and NP and TORCH
        if len(state_des.shape) <= 1:
            self.state_des = self.state_des.reshape(1, -1) # CP and NP and TORCH
        
    def compute_state_cost(self, state):
        """
        Compute state-dependent cost
        """
        if len(state.shape) <= 1:
            state = state.reshape(1, -1)
        delta_x = self.state_des - state
        state_cost = self.METHOD.einsum("ij,kj,ik->i", delta_x, self.Q, delta_x) # dx^T @ Q @ dx
        state_cost += 1e10*self.METHOD.sum(state[:, 2] <= 0) # if len(state_cost.shape) == 1 else 1e6*(state[2] <= 0) # CRASH COST
        return state_cost

    def compute_control_cost(self, u):
        """
        Compute control-dependent cost

        Costs that I've tried:

        control_cost = self.METHOD.einsum("ij,kj,ik->i", self.U_SHAPE_ARR, self.SYSTEM_NOISE_INV, self.U_SHAPE_ARR) # / self.config.CF2X.MAX_RPM
        control_cost = self.METHOD.einsum("ij,kj,ik->i", self.U_SHAPE_ARR, self.SYSTEM_NOISE_INV, du) / self.config.CF2X.MAX_RPM**2
        control_cost = self.METHOD.einsum("ij,kj,ik->i", self.U_SHAPE_ARR, self.SYSTEM_NOISE_INV, du)

        COST = GAMMA * ((u_tm1 - NOMINAL_U) @ SYSTEM_NOISE @ dU))
        """
        u_tm1, du = u
        # u_tm1 = (u_tm1 - self.config.CF2X.HOVER_RPM) / self.config.CF2X.MAX_RPM
        u_tm1 = u_tm1 / self.config.CF2X.MAX_RPM
        du = du / self.config.CF2X.MAX_RPM
        # du = self.METHOD.abs(du) # This is optional??
        self.U_SHAPE_ARR = self.METHOD.broadcast_to(u_tm1, self.U_SHAPE_ARR.shape)
        control_cost = self.METHOD.einsum("ij,kj,ik->i", self.U_SHAPE_ARR, self.SYSTEM_NOISE_INV, du)
        return self.config.TEMPERATURE * control_cost

    def __call__(self, state, u):
        return self.compute_state_cost(state) + self.compute_control_cost(u)
    

    