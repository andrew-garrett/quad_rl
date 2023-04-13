#################### IMPORTS ####################
#################################################


import numpy as np

from scipy.spatial.transform import Rotation 

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
        """
        Approximate next state given current state and control
        """
        new_state = self.postprocess(self.step_dynamics(self.preprocess(state, u)))
        return new_state



class AnalyticalModel(DynamicsModel): 
    """
    Child class for simple analytical dynamics model
    ALL EQUATION NUMBERS REFER TO MEAM 620 Proj1_1 HANDOUT (Uploaded to Drive)
    """
    def __init__(self, config):
        super().__init__(config)

    def motorModel(self, w_i):
        """Relate angular velocities [rpm] of motors to motor forces [N] and toqrues [N-m] via simplified motor model"""
        #Clip RPM to be in feasible range 
        F_i = self.config.CF2X.KF * np.clip(w_i, 0, self.config.CF2X.MAX_RPM)**2 #[N], eq (6)
        M_i = self.config.CF2X.KM * np.clip(w_i, 0, self.config.CF2X.MAX_RPM)**2 #[N/M], eq (7)
        return F_i, M_i

    def preprocess(self, state, u):
        """Go from control in terms or RPM to control u_1 (eq 10) and u_2 (eq 16)"""
        F, M = self.motorModel(u) #get forces and moments from motor model
        u1 = np.sum(F) #eq 10
        u2 = np.array([self.config.CF2X.L*(F[1] - F[4]), self.config.CF2X.L*(F[2] - F[0]), M[0] - M[1] + M[2] - M[3]]) #Eq (14, 16)
        return state, u1, u2 

    def step_dynamics(self, input):
        """
        Given the current state and control input u, use dynamics to find the accelerations
        Two coupled second order ODES 
        """
        state, u1, u2 = input # Decompose input
        xyz, rpy, velo, omega = state[:3], state[3:6], state[6:9], state[9:] #Decompose state
        F_ti, M_i = self.motorModel(u) #Get forces and moments due to control 

        #Coordinate transformation from drone frame to world frame, can use inverse of orientation
        w_R_d = Rotation.from_euler(('xyz'), rpy).inv().as_matrix() 

        # ---- Position, F = m*a ----
        f_g = self.config.CF2X.GRAVITY #force due to gravity 
        f_thrust = w_R_d @ np.array([0, 0, u1]) #force due to thrust, rotated into world frame

        #NO EXTERNAL FORCES (DRAG, DOWNWASH, GROUND EFFECT ETV FOR NOT)#TODO
        F_sum = f_g + f_thrust #net force [N]
        accel = F_sum/self.config.CF2X.M #solve eq 17 for for accel [m/s^2
        # ---- Orientation ------
        #Solving equation 18 for pqr dot 
        omega_dot = self.config.CF2X.J_inv @ (u2 - np.cross(omega, self.config.CF2X.J @ omega))

        return state, accel, omega_dot
    
    def postprocess(self, output):
        """Given the current state, control input, and time interval, propogate the state kinematics"""
        #Decompose output and state
        state, accel, omega_dot = output
        xyz, rpy, velo, omega = state[:3], state[3:6], state[6:9], state[9:]
        dt = self.config.DT
        
        #Apply kinematic equations (x_t = x_0 + v_0 dt + 0.5 a t^2, v_t = v_0 + a dt)
        xyz_dt = xyz + velo * dt + 0.5 * accel * dt**2
        velo_dt = velo + accel * dt
        
        #same for rotation 
        rpy_dt = rpy + omega * dt + 0.5 * omega_dot * dt**2
        omega_dt = omega + omega_dot * dt 
        
        #format in shape of state and return 
        return np.hstack((xyz_dt, rpy_dt, velo_dt, omega_dt))
