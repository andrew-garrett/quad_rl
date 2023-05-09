#################### IMPORTS ####################
#################################################


import os
import json
from copy import deepcopy
from typing import Any
try:
    import cupy as cp
except:
    print("Cupy not found, defaulting back to np/torch.")
import numpy as np
import torch
try:
    import pytorch3d.transforms as torch_transforms
except:
    pass
from scipy.spatial.transform import Rotation

#################### FUNCTIONAL DYNAMICS MODELS ####################
####################################################################

class AnalyticalModel: 
    """
    Class for simple analytical dynamics model
    PULL FROM DYNAMICS CLASS FROM BASE AVIARY CLASS AND MEAM 620 Proj 1_1 Handout
    DRONE SETUP SLIGHTLY DIFFERENT THAN 620 HANDOUT, equation numbers relate to handout
    Line numbers refer to BaseAviary Class

    """
    def __init__(self, config, explicit: bool=True):
        self.config = config
        self.explicit = explicit

    def motorModel(self, w_i):
        """Relate angular velocities [rpm] of motors to motor forces [N] and toqrues [N-m] via simplified motor model"""
        F_i = self.config.CF2X.KF*w_i**2 # [N], eq (6)
        M_i = self.config.CF2X.KM*w_i**2 # [N/m], eq (7)
        return F_i, M_i

    def preprocess(self, state, u):
        """Go from control in terms or RPM to control u_1 (eq 10) and u_2 (eq 16)"""
        F, M = self.motorModel(u)     # Forces, Moments from motor model
        u1 = np.sum(F, axis=1)        # Total Thrust, eq 10

        # Body Torques, X-shaped drone 
        torque_x = self.config.CF2X.L/np.sqrt(2) * (F[:, 0] + F[:, 1] - F[:, 2] - F[:, 3])  # line 821
        torque_y = self.config.CF2X.L/np.sqrt(2) * (-F[:, 0] + F[:, 1] + F[:, 2] - F[:, 3]) # line 822
        torque_z = -1*M[:, 0] + M[:, 1] - M[:, 2] + M[:, 3]                                 # line 819
        u2 = np.vstack([torque_x, torque_y, torque_z]).T                                    # Body Torques, eq 16

        return state, u1, u2 

    def step_dynamics(self, input_proc):
        """
        Given the current state and control input u, use dynamics to find the accelerations
        Two coupled second order ODES 

        """
        state, u1, u2 = input_proc # Decompose input
        xyz, velo, rpy, rpy_rates = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:]
        # Coordinate transformation from drone frame to world frame, can use orientation
        d_R_w = Rotation.from_euler('xyz', rpy)

        ##### Accleration via net force
        f_g = np.array([0, 0, self.config.CF2X.GRAVITY]) # force due to gravity 
        f_thrust = d_R_w.apply(np.vstack((np.zeros_like(u1), np.zeros_like(u1), u1)).T) # force due to thrust, rotated into world frame
        F_sum = f_thrust - f_g # net force [N]
        accel = F_sum/self.config.CF2X.M #solve eq 17 for for accel [m/s^2
        # ---- Orientation ------
        # Solving equation 18 for pqr dot
        if not self.explicit:
            rpy_rates = d_R_w.inv().apply(rpy_rates)

        omega_dot = self.config.CF2X.J_INV @ (u2 - np.cross(rpy_rates, (self.config.CF2X.J @ rpy_rates.T).T)).T
        
        if not self.explicit:
            omega_dot = d_R_w.apply(omega_dot.T)
        else:
            omega_dot = omega_dot.T
        return state, d_R_w, accel, omega_dot
    
    def postprocess(self, output):
        """Given the current state, control input, and time interval, propogate the state kinematics"""
        #Decompose output and state
        state, d_R_w, accel, omega_dot = output
        xyz, velo, rpy, rpy_rates = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:]
        dt = self.config.DT
        
        #Apply kinematic equations (same way it is done in dynamics)
        velo_dt = velo + accel * dt
        xyz_dt = xyz + velo * dt
        
        #same for rotation 
        rpy_rates_dt = rpy_rates + omega_dot * dt

        if self.explicit:
            rpy_dt = rpy + rpy_rates * dt
        else: 
            rpy_dt = rpy + d_R_w.inv().apply(rpy_rates) * dt

        rpy_wrapped = deepcopy(rpy_dt)
        rpy_wrapped[:,[0,2]] = (rpy_wrapped[:,[0,2]] + np.pi) % (2 * np.pi) - np.pi
        rpy_wrapped[:,1] = (rpy_wrapped[:,1] + np.pi/2) % (np.pi) - np.pi/2
        
        #format in shape of state and return 
        return np.hstack((xyz_dt, velo_dt, rpy_wrapped, rpy_rates_dt))

    def accelerationLabels(self, state, u):
        """Helper function to compare with ground truth accelerations calculated using dv/dt
            Input the states, output the models acceleration predictions"""
        _, _, linear_accel, angular_accel = self.step_dynamics(self.preprocess(state, u))
        return np.hstack((linear_accel, angular_accel))
    
    def __call__(self, state, u):
        return self.postprocess(self.step_dynamics(self.preprocess(state, u)))




class TorchAnalyticalModel(AnalyticalModel):
    """
    Child Class of the AnalyticalModel Class, this is implemented in torch.
    """
    def __init__(self, config, explicit=True):
        super().__init__(config, explicit)
        self.J = torch.from_numpy(self.config.CF2X.J).to(device=self.config.DEVICE, dtype=self.config.DTYPE)
        self.J_INV = torch.from_numpy(self.config.CF2X.J_INV).to(device=self.config.DEVICE, dtype=self.config.DTYPE)
        ##### Pre allocate some arrays
        self.f_g = torch.tensor([0., 0., self.config.CF2X.GRAVITY]).to(device=self.config.DEVICE, dtype=self.config.DTYPE) # force due to gravity 
        self.f_g = self.f_g.reshape(1, 3)
        
        self.convention = "xyz"

    def preprocess(self, state: torch.tensor, u: torch.tensor):
        """
        Parameters:
            - state: size(K, X_SPACE)
            - u: size(K, U_SPACE)
        
        Returns:
            - state: torch.tensor, size(K, X_SPACE)
            - u1: torch.tensor, size(K, 1)
            - u2: torch.tensor, size(K, 3)

        """
        F, M = self.motorModel(u)    # get forces and moments from motor model
        u1 = torch.sum(F, dim=1)     # eq 10

        #X drone torques 
        torque_x = self.config.CF2X.L/np.sqrt(2) * (F[:, 0] + F[:, 1] - F[:, 2] - F[:, 3])    # line 821
        torque_y = self.config.CF2X.L/np.sqrt(2) * (-F[:, 0] + F[:, 1] + F[:, 2] - F[:, 3])   # line 822
        torque_z = -1*M[:, 0] + M[:, 1] - M[:, 2] + M[:, 3]                                   # line 819
        u2 = torch.vstack([torque_x, torque_y, torque_z]).T                                   # eq 16
        
        # # Wrap rpy
        # rpy_wrapped = deepcopy(state[:, 6:9])
        # rpy_wrapped = torch.flip(rpy_wrapped, (1,))
        # # Coordinate transformation from drone frame to world frame
        # self.d_R_w = euler_angles_to_matrix(rpy_wrapped, self.convention)
        # # self.d_R_w = torch.transpose(self.d_R_w, 1, 2)

        # assert torch.allclose(torch_transforms.matrix_to_euler_angles(self.d_R_w, self.convention), rpy_wrapped, rtol=1e-10, atol=1e-10)
        
        return state, u1, u2
    
    def step_dynamics(self, input_proc: tuple((torch.tensor, torch.tensor, torch.tensor))):
        """
        Given the current state and control input u, use dynamics to find the accelerations
        Two coupled second order ODES

        Note: Scipy Rotation class euler angle range for "xyz" convention [[-pi, pi], [-pi/2, pi/2], [-pi, pi]]
        As such, when passing this to pytorch3d transforms, need to rewrap the Y coordinate of rpy

        Parameters:
            - input_proc: output of the preprocess function
        """
        state, u1, u2 = input_proc # Decompose input
        
        # ---- Position, F = m*a ----
        f_thrust = torch.zeros((1, 3, 1)).to(device=self.config.DEVICE, dtype=self.config.DTYPE)
        f_thrust[:, -1, :] = u1

        # Wrap rpy
        rpy_wrapped = deepcopy(state[:, 6:9])
        # rpy_wrapped = torch.tensor(Rotation.from_euler("xyz", rpy_wrapped.cpu().numpy()).as_euler("XYZ")).to(device=self.config.DEVICE, dtype=self.config.DTYPE)
        # Coordinate transformation from drone frame to world frame
        self.d_R_w = euler_angles_to_matrix(rpy_wrapped, self.convention)

        f_thrust = torch.bmm(self.d_R_w, f_thrust).squeeze(-1)

        # NO EXTERNAL FORCES (DRAG, DOWNWASH, GROUND EFFECT ETV FOR NOT) #TODO
        f_net = f_thrust - self.f_g # net force [N]
        accel = f_net/self.config.CF2X.M # solve eq 17 for for accel [m/s^2]
        
        # ---- Orientation ------
        # Solving equation 18 for pqr dot
        rpy_rates = state[:, 9:]
        if not self.explicit:
            rpy_rates = torch.bmm(torch.transpose(self.d_R_w, 1, 2), rpy_rates.unsqueeze(-1)).squeeze(-1)
            omega_dot = self.J_INV @ (u2 - torch.cross(rpy_rates, (self.J @ rpy_rates.T).T)).T
            omega_dot = torch.bmm(self.d_R_w, omega_dot.unsqueeze(-1))
        else:
            omega_dot = self.J_INV @ (u2 - torch.cross(rpy_rates, (self.J @ rpy_rates.T).T)).T
            omega_dot = omega_dot.T

        return state, accel, omega_dot
    

    def postprocess(self, output):
        """Given the current state, control input, and time interval, propogate the state kinematics"""
        #Decompose output and state
        state, accel, omega_dot = output
        xyz, velo, rpy_rates = state[:, :3], state[:, 3:6], state[:, 9:]
        dt = self.config.DT
        
        #Apply kinematic equations (same way it is done in dynamics)
        velo_dt = velo + accel * dt
        xyz_dt = xyz + velo * dt
        #same for rotation 
        rpy_rates_dt = rpy_rates + omega_dot * dt

        if self.explicit:
            omega_R = torch_transforms.axis_angle_to_matrix(rpy_rates * dt)
            rpy_dt = matrix_to_euler_angles(torch.bmm(omega_R, self.d_R_w), self.convention)
        else:
            omega_R = torch_transforms.axis_angle_to_matrix(rpy_rates * dt)
            rpy_dt = matrix_to_euler_angles(torch.bmm(omega_R, torch.transpose(self.d_R_w, 1, 2)), self.convention)

        # # wrap rpy_dt
        # rpy_wrapped = deepcopy(rpy_dt)
        # rpy_wrapped[:, [0, 2]] = (rpy_wrapped[:, [0, 2]] + torch.pi) % (2 * torch.pi) - torch.pi
        # rpy_dt[:, 1] = (rpy_dt[:,1] + torch.pi) % (np.pi) - torch.pi/2
        
        #format in shape of state and return 
        return torch.hstack((xyz_dt, velo_dt, rpy_dt, rpy_rates_dt))

    def accelerationLabels(self, state, u):
        """Helper function to compare with ground truth accelerations calculated using dv/dt
            Input the states, output the models acceleration predictions"""
        _, linear_accel, angular_accel = self.step_dynamics(self.preprocess(state, u))
        return torch.hstack((linear_accel, angular_accel))