#################### IMPORTS ####################
#################################################


import time
import numpy as np
from scipy.spatial.transform import Rotation

import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from bootstrap.utils import get_tracking_config, render_markers


#################### GLOBAL VARIABLES ####################
##########################################################  


# config.DRONE_MODEL
# config.PHYSICS
# config.VISION
# config.GUI
# config.RECORD
# config.PLOT
# config.USER_DEBUG_GUI
# config.AGGREGATE
# config.AGGREGATE_PHY_STEPS
# config.OBSTACLES
# config.SIMULATION_FREQ_HZ
# config.CONTROL_FREQ_HZ
# config.COLAB

# DEFAULT_DRONES = DroneModel("cf2x") # gym-pybullet-drones model
# DEFAULT_NUM_DRONES = 1
# DEFAULT_PHYSICS = Physics("pyb") # Physics("pyb_gnd_drag_dw")
# DEFAULT_VISION = False
# DEFAULT_GUI = True
# DEFAULT_RECORD_VISION = False
# DEFAULT_PLOT = True
# DEFAULT_USER_DEBUG_GUI = True
# DEFAULT_AGGREGATE = True
# DEFAULT_OBSTACLES = False
# DEFAULT_SIMULATION_FREQ_HZ = 240
# DEFAULT_CONTROL_FREQ_HZ = 48
# DEFAULT_OUTPUT_FOLDER = 'results'
# DEFAULT_COLAB = False
# DEFAULT_H = 10.0
# DEFAULT_OFFSET = 1.0


def initialize_tracking(trajectory, config):
    """
    Set up a Gym-Pybullet-Drones Env
    """
    # NxNxN cube configuration of drones, offset from each other in x, y, z by OFFSET and at z_min = H > 0
    cube_dim = trajectory.config.num_drones
    num_drones = trajectory.config.num_drones**3
    INIT_XYZS = config.OFFSET*np.indices((cube_dim, cube_dim, cube_dim)).reshape((3, num_drones)).T
    INIT_XYZS[:, -1] += config.H
    
    # Set initial orientation if parameterized in the config
    if "rpy0" in trajectory.config._fields:
        if trajectory.config["ax"] == "r":
            INIT_RPYS = np.array([[trajectory.config.rpy0*np.pi/180., 0., 0.] for k in range(num_drones)])
        elif trajectory.config["ax"] == "p":
            INIT_RPYS = np.array([[0., trajectory.config.rpy0*np.pi/180., 0.] for k in range(num_drones)])
        else:
            INIT_RPYS = np.array([[0., 0., trajectory.config.rpy0*np.pi/180.] for k in range(num_drones)])
    else:
        INIT_RPYS = np.array([[0., 0., np.pi*(k / num_drones)] for k in range(num_drones)])

    #### Create the environment with or without video capture ##
    if config.VISION: 
        env = VisionAviary(drone_model=config.DRONE_MODEL,
                           num_drones=num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=config.PHYSICS,
                           neighbourhood_radius=10,
                           freq=config.SIMULATION_FREQ_HZ,
                           aggregate_phy_steps=config.AGGREGATE_PHY_STEPS,
                           gui=config.GUI,
                           record=config.RECORD,
                           obstacles=config.OBSTACLES,
                           user_debug_gui=config.USER_DEBUG_GUI
                           )
    else: 
        env = CtrlAviary(drone_model=config.DRONE_MODEL,
                         num_drones=num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=config.PHYSICS,
                         neighbourhood_radius=10,
                         freq=config.SIMULATION_FREQ_HZ,
                         aggregate_phy_steps=config.AGGREGATE_PHY_STEPS,
                         gui=config.GUI,
                         record=config.RECORD,
                         obstacles=config.OBSTACLES,
                         user_debug_gui=config.USER_DEBUG_GUI
                        )
        
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=int(config.SIMULATION_FREQ_HZ/config.AGGREGATE_PHY_STEPS),
        num_drones=env.NUM_DRONES,
        output_folder=config.OUTPUT_FOLDER,
        colab=config.COLAB
    )

    #### Initialize the controllers ############################
    ctrl = [DSLPIDControl(drone_model=config.DRONE_MODEL) for k in range(env.NUM_DRONES)]

    return env, logger, ctrl


def get_control(trajectory, env, ctrl, config, target_state, obs, action):
    
    """
    Compute the controls in RPM
    """
    # Set targets
    target_pos = env.INIT_XYZS + target_state["x"]
    target_vel = np.zeros_like(env.INIT_XYZS) + target_state["x_dot"]
    target_rpy = env.INIT_RPYS # TODO: @Andrew, generate target roll and pitch from acceleration?
    target_rpy_rates = np.zeros_like(env.INIT_XYZS) # TODO: @Andrew, generate target roll and pitch rates from jerk?

    # # Apply noise to the targets
    # pos_noise = config.TARGET_NOISE_MODEL.rng.normal(loc=0., scale=config.TARGET_NOISE_MODEL.sigma_p, size=(env.NUM_DRONES, 3))
    # target_pos += pos_noise
    # vel_noise = config.TARGET_NOISE_MODEL.rng.normal(loc=0., scale=config.TARGET_NOISE_MODEL.sigma_v, size=(env.NUM_DRONES, 3))
    # target_vel += vel_noise
    # rpy_noise = config.TARGET_NOISE_MODEL.rng.normal(loc=0., scale=config.TARGET_NOISE_MODEL.sigma_r, size=(env.NUM_DRONES, 3))
    # target_rpy = (Rotation.from_euler("xyz", rpy_noise, degrees=False) * Rotation.from_euler("xyz", target_rpy, degrees=False)).as_euler("xyz", degrees=False)
    
    # Collect Position Error
    pos_error = np.zeros((env.NUM_DRONES, 3))
    for k in range(env.NUM_DRONES):
        action[str(k)], pos_error[k], _ = ctrl[k].computeControlFromState(control_timestep=config.CONTROL_PERIOD_STEPS*env.TIMESTEP,
                                                                          state=obs[str(k)]["state"],
                                                                          target_pos=target_pos[k],
                                                                          target_vel=target_vel[k],
                                                                          target_rpy=target_rpy[k],
                                                                          target_rpy_rates=target_rpy_rates[k]
                                                                         )
    return target_pos, target_vel, target_rpy, target_rpy_rates, action, pos_error


#################### RUNNER ####################
################################################


def track(trajectory, verbose=False):
    
    #### Initialize the simulation #############################
    config = get_tracking_config(trajectory=trajectory)

    env, logger, ctrl = initialize_tracking(trajectory, config)

    if np.cbrt(env.NUM_DRONES) <= 2 and config.GUI: #### Render Waypoints
        config = get_tracking_config(trajectory=trajectory)
        render_markers(env, config, points=trajectory.points)

    #### Run the simulation ####################################
    action = {str(k): np.array([0,0,0,0]) for k in range(env.NUM_DRONES)}
    t_counter = 0
    t_start = time.time()
    while t_counter < int(config.T_HORIZON*env.SIM_FREQ):
        
        #### Step the simulation ###########################################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ######################################
        if t_counter%config.CONTROL_PERIOD_STEPS == 0:
            target_state = trajectory.update(t_counter*env.TIMESTEP)
            target_pos, target_vel, target_rpy, target_rpy_rates, action, pos_error = get_control(trajectory, env, ctrl, config, target_state, obs, action)
            if np.cbrt(env.NUM_DRONES) <= 2 and config.GUI: #### Plot Trajectory and Flight
                render_markers(env, config, obs=obs, target_pos=target_pos)
            #### First check if we have finished our trajectory #############################
            #### namely, current position <= 0.05m and current speed <= 0.05m/s #############
            if trajectory.is_done and np.mean(np.linalg.norm(pos_error, axis=1)) <= 5e-2:
                if np.mean(np.linalg.norm(env.vel.copy(), axis=1)) <= 5e-2:
                    break
            #### Then check if it looks like we are gonna crash ############################
            elif np.any(env.pos.copy()[:, 2] <= config.OFFSET):
                break
        if env.PHYSICS.name == "dyn":
            rpy_rates = env.rpy_rates.copy()
        for k in range(env.NUM_DRONES): #### Log the simulation
            #### If we are using the explicit dynamics model, set the angular velocity in the state to be the rpy_rates
            if env.PHYSICS.name == "dyn":
                obs[str(k)]["state"][13:16] = rpy_rates[k]
            logger.log(
                drone=k,
                timestamp=t_counter/env.SIM_FREQ,
                state=obs[str(k)]["state"],
                control=np.hstack([target_pos[k], target_rpy[k], target_vel[k], target_rpy_rates[k]]),
                flat_trajectory=np.hstack([target_pos[k], target_vel[k], target_state["x_ddot"], target_state["x_dddot"], target_state["x_ddddot"], target_state["yaw"], target_state["yaw_dot"]])
            )
        if verbose and t_counter%env.SIM_FREQ == 0: #### Printout
            env.render()
            if config.VISION: #### Print matrices with the images captured by each drone 
                for k in range(env.NUM_DRONES):
                    print(obs[str(k)]["rgb"].shape, np.average(obs[str(k)]["rgb"]),
                          obs[str(k)]["dep"].shape, np.average(obs[str(k)]["dep"]),
                          obs[str(k)]["seg"].shape, np.average(obs[str(k)]["seg"])
                    )

        if config.GUI: #### Sync the simulation
            sync(t_counter, t_start, env.TIMESTEP)

        t_counter += config.AGGREGATE_PHY_STEPS # Propragate physics
    env.close() #### Close the environments
    logger.save() #### Save the simulation results