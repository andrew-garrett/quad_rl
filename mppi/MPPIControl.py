"""
The MPPIControl class inherits the BaseControl class

Basically, we need to migrate the functionality of MPPI_Node.py,
while also restructuring to fit the necessary methods for the BaseControl class.
"""

from copy import deepcopy
import numpy as np
import pybullet as p

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

from mppi.MPPINode import get_mppi_config, MPPI

class MPPIControl(DSLPIDControl):
    """
    MPPI control class for Crazyflies.
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 physics_model: str="dyn"
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        self.physics_model = physics_model
        # Create an MPPI Node
        self.mppi_config = get_mppi_config()
        self.mppi_node = MPPI(self.mppi_config, physics_model=physics_model)
        self.warmup_iters = self.mppi_config.T
        self.cur_state = None
        self.reference_trajectory = None

        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        """
        super().reset()

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the MPPI control action (as RPMs) for a single drone.


        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        ##### Get the current state
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        cur_state = np.hstack((cur_pos, cur_vel, cur_rpy, cur_ang_vel)).reshape(1, -1)
        self.cur_state = cur_state

        ##### Warm up the MPPI Controller
        while self.control_counter < self.warmup_iters:
            self.mppi_node.command(cur_state, False)
            self.control_counter += 1
        if self.control_counter == self.warmup_iters:
            self.warmup_iters = 0
            self.control_counter = 0

        ##### Get next control input
        rpm = self.mppi_node.command(cur_state, True)
        self.control_counter += 1
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        euler_e = target_rpy - cur_rpy
        ang_vel_e = target_rpy_rates - cur_ang_vel

        return rpm, pos_e, euler_e[-1]
    
    ################################################################################

    def set_reference_trajectory(self, t, trajectory, target_pos_i, target_vel_i, target_rpy_i, target_rpy_rates_i):
        """
        Sets the reference trajectory for MPPI to track.  If a reference trajectory has not yet been set,
        this function will iterate through the global trajectory for a full time-horizon and create one.
        After calling this function, self.reference_trajectory will be an np array of shape (self.mppi_config.T+1, self.mppi_config.X_SPACE)
        which represents a fully rolled out global trajectory.  This function also sets the self.curr_state to the 
        first state in the global trajectory if it has not yet been set, which only happens on the first rollout.

        Parameters:
        ----------
        t : float
            The current timestep of the global trajectory
        target_pos_i : ndarray
            (3,1)-shaped array of floats containing the initial desired position.
        target_rpy_i : ndarray
            (3,1)-shaped array of floats containing the initial desired orientation as roll, pitch, yaw.
        target_vel_i : ndarray
            (3,1)-shaped array of floats containing the initial desired velocity.
        target_rpy_rates_i : ndarray
            (3,1)-shaped array of floats containing the initial desired roll, pitch, and yaw rates.

        """
        if self.reference_trajectory is None:
            # Unroll the full trajectory for all timesteps in the time horizon
            self.reference_trajectory = np.zeros((self.mppi_config.T+1, self.mppi_config.X_SPACE))
            self.reference_trajectory[0] = np.hstack((target_pos_i, target_vel_i, target_rpy_i, target_rpy_rates_i))
            for t_ind in range(1, self.mppi_config.T):
                t_ref = t + t_ind*self.mppi_config.DT
                state_t = trajectory.update(t_ref)
                # xyz, velo, rpy, rpy_rates = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:], all in world frame except rpy_rates
                self.reference_trajectory[t_ind] = np.hstack((state_t["x"] + target_pos_i, state_t["x_dot"], target_rpy_i, target_rpy_rates_i))
        
        # We only need to do a full unroll once; after that, we can just roll it and add the state at the horizon
        t_ref = t + self.mppi_config.T*self.mppi_config.DT
        state_t = trajectory.update(t_ref)
        self.reference_trajectory[-1] = np.hstack((state_t["x"] + target_pos_i, state_t["x_dot"], target_rpy_i, target_rpy_rates_i))
        # TODO: This may be BUGGY
        self.reference_trajectory = np.roll(self.reference_trajectory, -1, 0)
        if self.cur_state is None:
            self.cur_state = self.reference_trajectory[-1]
        self.mppi_node.reset(desired_traj=deepcopy(self.reference_trajectory[:-1]))

    ################################################################################

    def get_trajectories(self, reference=False, rollout=False):
        """
        Gets the current global reference trajectory, and the current optimal rollout

        Parameters:
        ----------
        reference: bool
            if true, get the reference trajectory
        rollout: bool
            if true, get the rollout trajectory

        Returns:
        ----------
        (ref_traj, rollout_traj): tuple(np.array(T+1, X_SPACE), np.array(T+1, X_SPACE)
            the reference and rollout trajectories

        """
        ref_traj = None
        rollout_traj = None
        if reference:
            ref_traj = self.reference_trajectory[:-1]
        if rollout:
            rollout_traj = np.zeros((self.mppi_config.T+1, self.mppi_config.X_SPACE))
            rollout_traj[0] = deepcopy(self.cur_state)
            for t_ind in range(1, self.mppi_config.T+1):
                x_tm1 = deepcopy(rollout_traj[t_ind-1]).reshape(1, -1)
                u_tm1 = deepcopy(self.mppi_node.U[t_ind-1]).reshape(1, -1)
                rollout_traj[t_ind] = self.mppi_node.F(x_tm1, u_tm1).flatten()
            rollout_traj = np.roll(rollout_traj, -1, 0)[:-1]
        return (ref_traj, rollout_traj)
