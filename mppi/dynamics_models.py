#################### IMPORTS ####################
#################################################


import numpy as np


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


class DynamicsModel:
    def __init__(self, config):
        self.config = config

    def preprocess(self, state, u):
        """
        Preprocess state and control for network input
        """
        if len(state.shape) == 2:
            preprocessed = np.hstack((np.sin(state[:, 3:6]), np.cos(state[:, 3:6]), state[:, 6:], u))
        else:
            preprocessed = np.hstack((np.sin(state[3:6]), np.cos(state[3:6]), state[6:], u))
        return preprocessed

    def step_dynamics(self, input):
        """
        Compute linear and angular acceleration using dynamics network
        """
        acc = np.zeros((self.config.K, 6), dtype=self.config.DTYPE)
        return acc

    def postprocess(self, output):
        """
        Postprocess network outputs to obtain the resulting state in the state-space
        """
        new_state = np.random.standard_normal(size=(self.config.K, self.config.X_SPACE))
        return new_state

    def __call__(self, state, u):
        return self.postprocess(self.step_dynamics(self.preprocess(state, u)))