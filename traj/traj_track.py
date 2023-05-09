#################### IMPORTS ####################
#################################################


import time
import numpy as np
from scipy.spatial.transform import Rotation

import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from CustomLogger import CustomLogger
from gym_pybullet_drones.utils.utils import sync

from utils import get_tracking_config, render_markers, render_rollouts
from mppi.MPPIControl import MPPIControl


#################### GLOBAL VARIABLES ####################
##########################################################  


# config.DRONE_MODEL
# config.PHYSICS
# config.VISION
# config.GUI
# config.RECORD
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
        INIT_RPYS = np.array([[0., 0., 2.*np.pi*(k / num_drones) - np.pi + np.pi/4.] for k in range(num_drones)])

    ##### Create the environment with or without video capture
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
                           user_debug_gui=config.USER_DEBUG_GUI,
                           output_folder=config.OUTPUT_FOLDER
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
                         user_debug_gui=config.USER_DEBUG_GUI,
                         output_folder=config.OUTPUT_FOLDER
                        )
        
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    if config.LOG:
        logger = CustomLogger(
            logging_freq_hz=int(config.SIMULATION_FREQ_HZ/config.AGGREGATE_PHY_STEPS),
            num_drones=env.NUM_DRONES,
            output_folder=config.OUTPUT_FOLDER,
            colab=config.COLAB
        )
    else:
        logger = None

    #### Initialize the controllers ############################
    if config.CONTROLLER.lower() in ("mppi", "hybrid"):
        ctrl = [MPPIControl(drone_model=config.DRONE_MODEL, physics_model=config.PHYSICS.value) for k in range(env.NUM_DRONES)]
    else:
        ctrl = [DSLPIDControl(drone_model=config.DRONE_MODEL) for k in range(env.NUM_DRONES)]

    return env, logger, ctrl, PYB_CLIENT


def get_control(t, trajectory, env, ctrl, config, target_state, obs, action):
    """
    Compute the controls in RPM
    """
    # Set targets
    target_pos = env.INIT_XYZS + target_state["x"]
    target_vel = np.zeros_like(env.INIT_XYZS) + target_state["x_dot"]
    target_rpy = env.INIT_RPYS + np.array([0., 0., target_state["yaw"]])
    target_rpy_rates = np.zeros_like(env.INIT_XYZS) + np.array([0., 0., target_state["yaw_dot"]])

    # If using MPPI control, collect the next_time_horizon of state trajectories
    if config.CONTROLLER.lower() in ("mppi", "hybrid"):
        ref_traj_arr, rollout_traj_arr = [], []
        for k in range(env.NUM_DRONES):
            ctrl[k].set_reference_trajectory(t, trajectory, env.INIT_XYZS[k], target_vel[k], target_rpy[k], target_rpy_rates[k])
            # Rollout and reference trajectory visualization is extremely time-consuming, so it's best to avoid it
            if np.cbrt(env.NUM_DRONES) <= 2 and config.USER_DEBUG_GUI:
                ref_traj_k, rollout_traj_k = ctrl[k].get_trajectories(reference=True, rollout=True)
                if ref_traj_k is not None:
                    ref_traj_arr.append(ref_traj_k)
                if rollout_traj_k is not None:
                    rollout_traj_arr.append(rollout_traj_k)
        if np.cbrt(env.NUM_DRONES) <= 2 and config.USER_DEBUG_GUI:
            config = render_rollouts(config, ref_traj_arr=ref_traj_arr, rollout_traj_arr=rollout_traj_arr)
    
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
    return config, target_pos, target_vel, target_rpy, target_rpy_rates, action, pos_error


#################### RUNNER ####################
################################################


def track(trajectory, config_fpath="./configs/tracking/tracking_config.json", verbose=False):

    #### Initialize the simulation #############################
    config = get_tracking_config(trajectory=trajectory, config_fpath=config_fpath)
    env, logger, ctrl, pyb_client = initialize_tracking(trajectory, config)

    if np.cbrt(env.NUM_DRONES) <= 2 and config.USER_DEBUG_GUI: #### Render Waypoints
        config = get_tracking_config(trajectory=trajectory, config_fpath=config_fpath)
        render_markers(env, config, points=trajectory.points)

    #### Run the simulation ####################################
    action = {str(k): env.HOVER_RPM*np.ones(4) for k in range(env.NUM_DRONES)}
    t_counter = 0
    t_start = time.time()
    while t_counter < int(config.T_FINISH*env.SIM_FREQ):
        
        #### Step the simulation ###########################################################
        obs, reward, done, info = env.step(action)

        #### If we are using the explicit dynamics model, set the angular velocity in the state to be the rpy_rates
        if env.PHYSICS.value == "dyn":
            rpy_rates = env.rpy_rates.copy()
            for k in range(env.NUM_DRONES):
                obs[str(k)]["state"][13:16] = rpy_rates[k]

        #### Compute control at the desired frequency ######################################
        if t_counter%config.CONTROL_PERIOD_STEPS == 0:
            target_state = trajectory.update(t_counter*env.TIMESTEP)

            config, target_pos, target_vel, target_rpy, target_rpy_rates, action, pos_error = get_control(t_counter*env.TIMESTEP, trajectory, env, ctrl, config, target_state, obs, action)
            if np.cbrt(env.NUM_DRONES) <= 2 and config.USER_DEBUG_GUI: #### Plot Trajectory and Flight
                render_markers(env, config, obs=obs, target_pos=target_pos)
            #### First check if we have finished our trajectory #############################
            #### namely, current position <= 0.05m and current speed <= 0.05m/s #############
            if trajectory.is_done and np.mean(np.linalg.norm(pos_error, axis=1)) <= 5e-2:
                if np.mean(np.linalg.norm(env.vel.copy(), axis=1)) <= 5e-2:
                    break
            #### Then check if it looks like we are gonna crash ############################
            elif np.any(env.pos.copy()[:, 2] <= config.OFFSET):
                break
        
        if config.RECORD:
            #### If we're recording, don't make videos that are too long
            if t_counter*env.TIMESTEP > 12.0:
                break
            #### Track the main drone with the camera
            env.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=2,
                                                                yaw=45,
                                                                pitch=-30,
                                                                roll=0,
                                                                cameraTargetPosition=obs["0"]["state"][:3],
                                                                upAxisIndex=2,
                                                                physicsClientId=pyb_client
                                                                )
        if config.LOG:
            try:
                for k in range(env.NUM_DRONES): #### Log the simulation
                    logger.log(
                        drone=k,
                        timestamp=t_counter/env.SIM_FREQ,
                        state=obs[str(k)]["state"],
                        control=np.hstack([target_pos[k], target_vel[k], target_rpy[k], target_rpy_rates[k]]),
                        optimal_trajectory=ctrl[k].optimal_rollout_next_state if hasattr(ctrl[k], "optimal_rollout_next_state") else np.hstack([target_pos[k], target_vel[k], target_rpy[k], target_rpy_rates[k]])
                    )
            except Exception as e:
                print(e)
        
        if verbose and t_counter%env.SIM_FREQ == 0: #### Printout
            env.render()
            if config.VISION: #### Print matrices with the images captured by each drone 
                for k in range(env.NUM_DRONES):
                    print(obs[str(k)]["rgb"].shape, np.average(obs[str(k)]["rgb"]),
                        obs[str(k)]["dep"].shape, np.average(obs[str(k)]["dep"]),
                        obs[str(k)]["seg"].shape, np.average(obs[str(k)]["seg"])
                    )

        if config.GUI: #### Sync the simulation
            sync(t_counter, t_start, config.CONTROL_PERIOD_STEPS*env.TIMESTEP)

        t_counter += config.AGGREGATE_PHY_STEPS # Propragate physics
    env.close()   ##### Close the environments
    if config.LOG:
        logger.save() ##### Save the simulation results (npy and png plots)