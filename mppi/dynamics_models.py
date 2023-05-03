#################### IMPORTS ####################
#################################################


try:
    import cupy as cp
except:
    print("Cupy not found, defaulting back to np/torch.")
import numpy as np
import os
import torch
import json
from copy import deepcopy

from scipy.spatial.transform import Rotation
import torch


from training.lightning import DynamicsLightningModule

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
        acc = np.zeros((self.config.K, 6))
        return acc

    def postprocess(self, output):
        """
        Postprocess network outputs to obtain the resulting state in the state-space
        """
        new_state = np.random.standard_normal(size=(self.config.K, self.config.X_SPACE))
        return new_state

    def __call__(self, state, u):
        """
        Approximate next state given current state and control
        """
        new_state = self.postprocess(self.step_dynamics(self.preprocess(state, u)))
        return new_state



class AnalyticalModel(DynamicsModel): 
    """
    Child class for simple analytical dynamics model
    PULL FROM DYNAMICS CLASS FROM BASE AVIARY CLASS AND MEAM 620 Proj 1_1 Handouy
    DRONE SETUP SLIGHTLY DIFFERENT THAN 620 HANDOUT, equation numbers relate to handout
    Line numbers refer to BaseAviary Class

    """
    def __init__(self, config, explicit = True):
        super().__init__(config)
        self.is_explicit = explicit 

    def motorModel(self, w_i):
        """Relate angular velocities [rpm] of motors to motor forces [N] and toqrues [N-m] via simplified motor model"""
        F_i = self.config.CF2X.KF*w_i**2 # [N], eq (6)
        M_i = self.config.CF2X.KM*w_i**2 # [N/M], eq (7)
        return F_i, M_i

    def preprocess(self, state, u):
        """Go from control in terms or RPM to control u_1 (eq 10) and u_2 (eq 16)"""
        F, M = self.motorModel(u)     #get forces and moments from motor model
        u1 = np.sum(F, axis=1)        #eq 10

        #X drone torques 
        torque_x = self.config.CF2X.L/np.sqrt(2) * (F[:, 0] + F[:, 1] - F[:, 2] - F[:, 3])  #line 821
        torque_y = self.config.CF2X.L/np.sqrt(2) * (-F[:, 0] + F[:, 1] + F[:, 2] - F[:, 3]) #line 822
        torque_z = -1*M[:, 0] + M[:, 1] - M[:, 2] + M[:, 3]                                 #line 819
        u2 = np.vstack([torque_x, torque_y, torque_z]).T                           #eq 16

        return state, u1, u2 

    def step_dynamics(self, input_proc):
        """
        Given the current state and control input u, use dynamics to find the accelerations
        Two coupled second order ODES 
        """
        state, u1, u2 = input_proc # Decompose input
        xyz, velo, rpy, rpy_rates = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:]
        # Coordinate transformation from drone frame to world frame, can use orientation
        d_R_w = Rotation.from_euler(('xyz'), rpy)

        # ---- Position, F = m*a ----
        f_g = np.array([0, 0, self.config.CF2X.GRAVITY]) #force due to gravity 
        f_thrust = d_R_w.apply(np.vstack((np.zeros_like(u1), np.zeros_like(u1), u1)).T) # force due to thrust, rotated into world frame

        #NO EXTERNAL FORCES (DRAG, DOWNWASH, GROUND EFFECT ETV FOR NOT) #TODO
        F_sum = f_thrust - f_g # net force [N]
        accel = F_sum/self.config.CF2X.M #solve eq 17 for for accel [m/s^2
        # ---- Orientation ------
        # Solving equation 18 for pqr dot
        omega_dot = self.config.CF2X.J_INV @ (u2 - np.cross(rpy_rates, (self.config.CF2X.J @ rpy_rates.T).T)).T
        return state, accel, omega_dot.T
    
    def postprocess(self, output):
        """Given the current state, control input, and time interval, propogate the state kinematics"""
        #Decompose output and state
        state, accel, omega_dot = output
        xyz, velo, rpy, rpy_rates = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:]
        dt = self.config.DT
        
        #Apply kinematic equations (same way it is done in dynamics)
        velo_dt = velo + accel * dt
        xyz_dt = xyz + velo_dt * dt
        
        #same for rotation 
        rpy_rates_dt = rpy_rates + omega_dot * dt
        rpy_dt = rpy + rpy_rates_dt * dt

        rpy_wrapped = deepcopy(rpy_dt)
        rpy_wrapped[0,[0,2]] = (rpy_wrapped[0,[0,2]] + np.pi) % (2 * np.pi) - np.pi
        rpy_wrapped[0,1] = (rpy_wrapped[0,1] + np.pi/2) % (np.pi) - np.pi/2
        
        #format in shape of state and return 
        return np.hstack((xyz_dt, velo_dt, rpy_wrapped, rpy_rates_dt))

    def accelerationLabels(self, state, u):
        """Helper function to compare with ground truth accelerations calculated using dv/dt
            Input the states, output the models acceleration predictions"""
        _ , linear_accel, angular_accel = self.step_dynamics(self.preprocess(state, u))
        return np.hstack((linear_accel, angular_accel))



class SampleLearnedModel(DynamicsModel):
    """
    Sample Class for 
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.parse_config()

        self.nn.eval()
        self.nn_gt_min = torch.load(os.path.join(self.dataset_path, "train_nn_gt_min.pt"))
        self.nn_gt_max = torch.load(os.path.join(self.dataset_path, "train_nn_gt_max.pt"))
        self.state_min = torch.load(os.path.join(self.dataset_path, "train_state_min.pt"))
        self.state_max = torch.load(os.path.join(self.dataset_path, "train_state_max.pt"))
    
    def parse_config(self):
        self.dataset_path = os.path.join(self.config.NN_DATASET_ROOT, self.config.NN_DATASET_NAME, "torch_dataset")
        self.nn_ckpt_path = self.config.NN_CKPT_PATH
        self.training_config_path = self.config.NN_CONFIG_PATH
        with open(self.training_config_path, "r") as f:
            self.training_config = json.load(f)
        self.nn = DynamicsLightningModule.load_from_checkpoint(self.nn_ckpt_path, config=self.training_config, log_dir=None)
        self.max_rpm = self.config.CF2X.MAX_RPM
    
    def normalize(self, val, minval, maxval):
        norm = (val - minval)/(maxval - minval)
        norm = norm*2 - 1
        return norm
    
    def denormalize(self, val, minval, maxval):
        denorm = (val + 1)/2
        denorm = (denorm*(maxval - minval)) + minval
        return denorm

    def preprocess(self, state, u):
        """
        Preprocess state and control for network input
        (normalize inputs, preprocess angles w/ sin and cos etc)
        """
        #if state.shape[0] == 1:
        xyz, velo, rpy, rpy_rates = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:]

        ## DIVIDE U BY MAX RPM
        u /= self.max_rpm

        nn_input = np.concatenate([np.sin(rpy), np.cos(rpy), velo, rpy_rates, u], axis=1)
        nn_input = torch.tensor(nn_input)
        nn_input_normalized = self.normalize(nn_input, self.state_min, self.state_max)
        return state, nn_input_normalized #preprocessed

    def step_dynamics(self, input):
        """
        Compute Accelerations OR Delta State using trained Weights/Biases of Network
        (Carry out forward pass of network)
        """
        state, nn_input = input
        with torch.no_grad():
            nn_output = self.nn.model(nn_input.float(), None)
        return state, nn_output

    def postprocess(self, output):
        """Given the current state, control input, and time interval, propogate the state kinematics"""
        #Decompose output and state
        state, nn_output = output
        nn_output_denormalized = self.denormalize(nn_output, self.nn_gt_min, self.nn_gt_max).numpy()
        accel, omega_dot = (nn_output_denormalized[:,:3], nn_output_denormalized[:,3:])
        xyz, velo, rpy, rpy_rates = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:]
        dt = self.config.DT
        
        #Apply kinematic equations (same way it is done in dynamics)
        velo_dt = velo + accel * dt
        xyz_dt = xyz + velo_dt * dt
        
        #same for rotation 
        rpy_rates_dt = rpy_rates + omega_dot * dt 
        rpy_dt = rpy + rpy_rates_dt * dt

        # wrap rpy_dt
        rpy_wrapped = deepcopy(rpy_dt)
        rpy_wrapped[0,[0,2]] = (rpy_wrapped[0,[0,2]] + np.pi) % (2 * np.pi) - np.pi
        rpy_wrapped[0,1] = (rpy_wrapped[0,1] + np.pi/2) % (np.pi) - np.pi/2
        
        #format in shape of state and return 
        return np.hstack((xyz_dt, velo_dt, rpy_wrapped, rpy_rates_dt))
    
    def accelerationLabels(self, state, u):
        """Helper function to compare with ground truth accelerations calculated using dv/dt
            Input the states, output the models acceleration predictions"""
        _ , nn_output = self.step_dynamics(self.preprocess(state, u))
        nn_output_denormalized = self.denormalize(nn_output, self.nn_gt_min, self.nn_gt_max).numpy()
        linear_accel, angular_accel = (nn_output_denormalized[:,:3], nn_output_denormalized[:,3:])

        return np.hstack((linear_accel, angular_accel))

class TorchAnalyticalModel(AnalyticalModel):
    """
    Child Class of the AnalyticalModel Class, this is implemented in torch.
    """
    def __init__(self, config, explicit=True):
        super().__init__(config, explicit)
        self.J = torch.from_numpy(self.config.CF2X.J).to(device=self.config.DEVICE, dtype=self.config.DTYPE)
        self.J_INV = torch.from_numpy(self.config.CF2X.J_INV).to(device=self.config.DEVICE, dtype=self.config.DTYPE)

    def preprocess(self, state, u):
        """Go from control in terms or RPM to control u_1 (eq 10) and u_2 (eq 16)"""
        F, M = self.motorModel(u) # get forces and moments from motor model
        u1 = torch.sum(F, dim=1) # eq 10

        #X drone torques 
        torque_x = self.config.CF2X.L/np.sqrt(2) * (F[:, 0] + F[:, 1] - F[:, 2] - F[:, 3])  # line 821
        torque_y = self.config.CF2X.L/np.sqrt(2) * (-F[:, 0] + F[:, 1] + F[:, 2] - F[:, 3]) # line 822
        torque_z = -1*M[:, 0] + M[:, 1] - M[:, 2] + M[:, 3]                                 # line 819
        u2 = torch.vstack([torque_x, torque_y, torque_z]).T                                 # eq 16
        return state, u1, u2 
    
    def step_dynamics(self, input_proc):
        """
        Given the current state and control input u, use dynamics to find the accelerations
        Two coupled second order ODES 
        """
        state, u1, u2 = input_proc # Decompose input
        xyz, velo, rpy, rpy_rates = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:]
        # Coordinate transformation from drone frame to world frame, can use orientation
        d_R_w = Rotation.from_euler(('xyz'), rpy.cpu())

        # ---- Position, F = m*a ----
        f_g = torch.tensor([0, 0, self.config.CF2X.GRAVITY]).to(device=self.config.DEVICE, dtype=self.config.DTYPE) #force due to gravity 
        u1 = u1.cpu()
        f_thrust = d_R_w.apply(torch.vstack((torch.zeros_like(u1), torch.zeros_like(u1), u1)).T)  #force due to thrust, rotated into world frame

        #NO EXTERNAL FORCES (DRAG, DOWNWASH, GROUND EFFECT ETV FOR NOT) #TODO
        F_sum = torch.tensor(f_thrust).to(device=self.config.DEVICE, dtype=self.config.DTYPE) - f_g # net force [N]
        accel = F_sum/self.config.CF2X.M #solve eq 17 for for accel [m/s^2]
        # ---- Orientation ------
        #Solving equation 18 for pqr dot 
        omega_dot = self.J_INV @ (u2 - torch.cross(rpy_rates, (self.J @ rpy_rates.T).T)).T
        return state, accel, omega_dot.T
    

    def postprocess(self, output):
        """Given the current state, control input, and time interval, propogate the state kinematics"""
        #Decompose output and state
        state, accel, omega_dot = output
        xyz, velo, rpy, rpy_rates = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:]
        dt = self.config.DT
        
        #Apply kinematic equations (same way it is done in dynamics)
        velo_dt = velo + accel * dt
        xyz_dt = xyz + velo_dt * dt
        
        #same for rotation 
        rpy_rates_dt = rpy_rates + omega_dot * dt
        rpy_dt = rpy + rpy_rates_dt * dt
        
        #format in shape of state and return 
        return torch.hstack((xyz_dt, velo_dt, rpy_dt, rpy_rates_dt))
