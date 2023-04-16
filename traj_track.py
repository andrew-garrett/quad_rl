#################### IMPORTS ####################
#################################################


import json
from collections import namedtuple
import time
import argparse
try:
    import cupy as cp
except:
    print("cupy not available, defaulting to numpy")
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


#################### TRACKING PARAMETERS ####################
#############################################################


def get_tracking_config(trajectory=None, config_fpath="./configs/tracking_config.json"):
    """
    Creates a config object for tracking parameters

    Parameters:
        - trajectory: TrajectoryGenerator or None - the trajectory object
        - config_fpath: str - the filepath of the desired config file

    Returns:
        - config: namedtuple() - the tracking config object
    """
    with open(config_fpath, "r") as f:
        config_dict =  json.load(f)

    # Set derived/processed/other parameters
    config_dict["DRONE_MODEL"] = DroneModel(config_dict["DRONE_MODEL"])
    config_dict["PHYSICS"] = Physics(config_dict["PHYSICS"])
    config_dict["AGGREGATE_PHY_STEPS"] = int(config_dict["SIMULATION_FREQ_HZ"]/config_dict["CONTROL_FREQ_HZ"]) if config_dict["AGGREGATE"] else 1
    config_dict["CONTROL_PERIOD_STEPS"] = int(np.floor(config_dict["SIMULATION_FREQ_HZ"]/config_dict["CONTROL_FREQ_HZ"]))
    config_dict["TARGET_NOISE_MODEL"]["rng"] = cp.random if "cp" in globals().keys() else np.random.default_rng()
    config_dict["TARGET_NOISE_MODEL"] = namedtuple("noise_model", config_dict["TARGET_NOISE_MODEL"].keys())(**config_dict["TARGET_NOISE_MODEL"])
    if trajectory is None:
        config_dict["OUTPUT_FOLDER"] = f"sim_data" # TODO: @Andrew 
        config_dict["T_HORIZON"] *= 1.1*0 # TODO: @Andrew 
    else:
        # Trajectory parameters overwrites tracking parameters
        config_dict["CONTROL_PERIOD_STEPS"] = int(np.floor(config_dict["SIMULATION_FREQ_HZ"]/config_dict["CONTROL_FREQ_HZ"])) # TODO: @Andrew 
        config_dict["OUTPUT_FOLDER"] = f"{trajectory.root}sim_data"
        config_dict["T_HORIZON"] = 1.1*trajectory.T_horizon # TODO: @Andrew
    if config_dict["GUI"]:
        try:
            config_dict["VIZ"] = get_viz()
        except:
            pass

    config = namedtuple("tracking_config", config_dict.keys())(**config_dict)
    return config

def get_viz():
    """
    Creates a config object for visualization parameters
    """
    sphereRadius = 0.05
    markers = {}
    markers["waypointSphereId"] = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, 
        rgbaColor=[1, 0, 0, 1],
        radius=sphereRadius
    )
    markers["flightSphereId"] = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, 
        rgbaColor=[0, 1, 0, 1],
        radius=sphereRadius*0.2
    )
    markers["trajSphereId"] = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, 
        rgbaColor=[0, 0, 1, 1],
        radius=sphereRadius*0.2
    )

    config = namedtuple("viz_config", markers.keys())(**markers)
    return config


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
    cube_dim = trajectory.config["num_drones"]
    num_drones = trajectory.config["num_drones"]**3
    INIT_XYZS = config.OFFSET*np.indices((cube_dim, cube_dim, cube_dim)).reshape((3, num_drones)).T
    INIT_XYZS[:, -1] += config.H
    
    # Set initial orientation if parameterized in the config
    if "rpy0" in trajectory.config.keys():
        if trajectory.config["ax"] == "r":
            INIT_RPYS = np.array([[trajectory.config["rpy0"]*np.pi/180., 0., 0.] for k in range(num_drones)])
        elif trajectory.config["ax"] == "p":
            INIT_RPYS = np.array([[0., trajectory.config["rpy0"]*np.pi/180., 0.] for k in range(num_drones)])
        else:
            INIT_RPYS = np.array([[0., 0., trajectory.config["rpy0"]*np.pi/180.] for k in range(num_drones)])
    else:
        INIT_RPYS = np.array([[0., 0., 2*np.pi*(k / num_drones)] for k in range(num_drones)])

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


def render_markers(env, config, points=None, obs=None, target_pos=None):
    """
    Render Waypoint, Trajectory, and Flight Path Markers in the Pybullet Environment
    """
    if points is not None: # Render larger spheres for Waypoints
        plt_num_wpts = min(100, points.shape[0])
        plt_wpt_inds = np.linspace(0, points.shape[0] - 1, num=plt_num_wpts).astype("int")
        for wpt_ind in plt_wpt_inds:
            for k in range(env.NUM_DRONES):
                p.createMultiBody(
                    baseMass=0,
                    baseInertialFramePosition=[0, 0, 0],
                    baseVisualShapeIndex=config.VIZ.waypointSphereId, 
                    basePosition=env.INIT_XYZS[k, :] + points[wpt_ind],
                    useMaximalCoordinates=1
                )
    if obs is not None: # Render smaller spheres for Trajectory and FLight Path
        for k in range(env.NUM_DRONES):
            p.createMultiBody(
                baseMass=0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=config.VIZ.trajSphereId, 
                basePosition=target_pos[k],
                useMaximalCoordinates=1
            )
            p.createMultiBody(
                baseMass=0,
                baseInertialFramePosition=[0, 0, 0],
                baseVisualShapeIndex=config.VIZ.flightSphereId, 
                basePosition=obs[str(k)]["state"][:3],
                useMaximalCoordinates=1
            )


def get_control(trajectory, env, ctrl, config, target_state, obs, action):
    
    """
    Compute the controls in RPM
    """
    # Set targets
    target_pos = env.INIT_XYZS + target_state["x"]
    target_vel = np.zeros_like(env.INIT_XYZS) + target_state["x_dot"].reshape(1, 3)
    target_rpy = np.array([[0., 0., 2*np.pi*(k / env.NUM_DRONES)] for k in range(env.NUM_DRONES)])
    # Apply noise to the target
    if not trajectory.is_done:
        pos_noise = config.TARGET_NOISE_MODEL.rng.normal(loc=0., scale=config.TARGET_NOISE_MODEL.sigma_p, size=(env.NUM_DRONES, 3))
        target_pos += pos_noise
        vel_noise = config.TARGET_NOISE_MODEL.rng.normal(loc=0., scale=config.TARGET_NOISE_MODEL.sigma_v, size=(env.NUM_DRONES, 3))
        target_vel += vel_noise
        rpy_noise = config.TARGET_NOISE_MODEL.rng.normal(loc=0., scale=config.TARGET_NOISE_MODEL.sigma_r, size=(env.NUM_DRONES, 3))
        target_rpy = (Rotation.from_euler("xyz", rpy_noise, degrees=False) * Rotation.from_euler("xyz", target_rpy, degrees=False)).as_euler("xyz", degrees=False)
    # Collect Position Error
    pos_error = np.zeros((env.NUM_DRONES, 3))
    for k in range(env.NUM_DRONES):
        action[str(k)], pos_error[k], _ = ctrl[k].computeControlFromState(control_timestep=config.CONTROL_PERIOD_STEPS*env.TIMESTEP,
                                                                          state=obs[str(k)]["state"],
                                                                          target_pos=target_pos[k],
                                                                          target_vel=target_vel[k],
                                                                          target_rpy=target_rpy[k]
                                                                         )
    return target_pos, target_vel, target_rpy, action, pos_error


#################### RUNNER ####################
################################################


def track(trajectory):
    
    #### Initialize the simulation #############################
    config = get_tracking_config(trajectory=trajectory)

    env, logger, ctrl = initialize_tracking(trajectory, config)

    if np.cbrt(env.NUM_DRONES) <= 2 and config.GUI: #### Render Waypoints
        config = get_tracking_config(trajectory=trajectory)
        render_markers(env, config, points=trajectory.points)

    #### Run the simulation ####################################
    action = {str(k): np.array([0,0,0,0]) for k in range(env.NUM_DRONES)}
    t_counter = 0
    START = time.time()

    while t_counter < int(config.T_HORIZON*env.SIM_FREQ):
        
        #### Step the simulation ###########################################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ######################################
        if t_counter%config.CONTROL_PERIOD_STEPS == 0:
            target_state = trajectory.update(t_counter*env.TIMESTEP)
            target_pos, target_vel, target_rpy, action, pos_error = get_control(trajectory, env, ctrl, config, target_state, obs, action)
            if np.cbrt(env.NUM_DRONES) <= 2 and config.GUI: #### Plot Trajectory and Flight
                render_markers(env, config, obs=obs, target_pos=target_pos)
            #### First check if we have finished our trajectory ############################
            #### namely, current position <= 0.05m and current speed <= 0.05m/s #############
            if trajectory.is_done and np.mean(np.linalg.norm(pos_error, axis=1)) <= 5e-2:
                if np.mean(np.linalg.norm(env.vel, axis=1)) <= 5e-2:
                    break
            #### Then check if it looks like we are gonna crash ############################
            elif np.any(env.pos[:, 2] <= config.OFFSET):
                break

        for k in range(env.NUM_DRONES): #### Log the simulation
            logger.log(
                drone=k,
                timestamp=t_counter/env.SIM_FREQ,
                state=obs[str(k)]["state"],
                control=np.hstack([target_pos[k], target_rpy[k], target_vel[k], np.zeros(3)])
            )
        if t_counter%env.SIM_FREQ == 0: #### Printout
            env.render()
            if config.VISION: #### Print matrices with the images captured by each drone 
                for k in range(env.NUM_DRONES):
                    print(obs[str(k)]["rgb"].shape, np.average(obs[str(k)]["rgb"]),
                          obs[str(k)]["dep"].shape, np.average(obs[str(k)]["dep"]),
                          obs[str(k)]["seg"].shape, np.average(obs[str(k)]["seg"])
                    )

        if config.GUI: #### Sync the simulation
            sync(t_counter, START, env.TIMESTEP)

        t_counter += config.AGGREGATE_PHY_STEPS # Propragate physics
    env.close() #### Close the environments
    logger.save() #### Save the simulation results
         

# if __name__ == "__main__":
    