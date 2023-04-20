#################### IMPORTS ####################
#################################################


import os
import shutil
from zipfile import ZipFile
import json
from collections import namedtuple
from itertools import product
try:
    import cupy as cp
except:
    print("cupy not available, defaulting to numpy")
import numpy as np
import matplotlib.pyplot as plt

import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from bootstrap.task_battery import TRAJECTORY_PARAMS, DEFAULT_TASK_NAME, DEFAULT_DATASET_NAME, DEFAULT_ROOT


######################## UTILITY FUNCTIONS ########################
###################################################################


def cleanup(root, dataset_name):
    """
    Zip a dataset and delete the uncompressed folder
    """
    data_root = os.path.join(root, dataset_name)
    ##### Copy the tracking config to the dataset root
    shutil.copy("./configs/tracking_config.json", data_root)
    os.rename(os.path.join(data_root, "tracking_config.json"), os.path.join(data_root, f"{dataset_name}_TRACKING_CONFIG.json"))
    ##### Walk the dataset directory and zip it's contents
    with ZipFile(f"{data_root}.zip", "w") as zip_object:
        for folder, sub_folders, f_names in os.walk(data_root):
            for f_name in f_names:
                f_path = os.path.join(folder, f_name)
                zip_object.write(f_path, f_path[len(data_root):])
    
    if os.path.exists(f"{data_root}.zip"):
        ##### Delete the uncompressed dataset folder
        shutil.rmtree(data_root)
        return True
    else:
        return False


######################## CONFIG/PARAMETER MANIPULATION ########################
###############################################################################


def get_task_param_grid(params):
    """
    Create a grid of all combinations of list-based parameters

    Parameters:
        - params: dict() - parameter dictionary for generating tasks
        
    """
    param_spaces = [v for k, v in sorted(params.items(), key=lambda x: x[0]) if k not in TRAJECTORY_PARAMS]
    return product(*param_spaces)


def get_traj_params(
        root=DEFAULT_ROOT, 
        dataset_name=DEFAULT_DATASET_NAME, 
        task_name=DEFAULT_TASK_NAME
    ):
    """
    Prepare parameters for planning a trajectory

    Parameters:
        - root: str - The root path for where datasets are located
        - dataset_name: str - The name of the dataset
        - task_name: str - The name of the current task
    
    Returns:
        - params_prepped: dict() - dictionary of (parameter_name, parameter_value) key-value pairs
            - Note: all values in the dictionary should be scalars or strings, no lists

    """

    # load params
    with open(os.path.join(root, dataset_name, f"{dataset_name}_TASK_CONFIG.json"), "r") as f:
        params_raw = json.load(f)

    params_prepped = {}
    for k in params_raw.keys():
        # find the taskcase params for task_name
        if k in task_name:
            params_prepped = params_raw[k]
            param_pairs = task_name[len(f"task_{k}_"):].split("_")
            for param_pair in param_pairs:
                # set list-based parameters to a scalar
                key, value = param_pair.split("-")
                try:
                    # handles numerical values
                    params_prepped[key] = float(value)
                except:
                    # handles string values
                    params_prepped[key] = value
            break
    
    return params_prepped


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
        # config_dict["CONTROL_PERIOD_STEPS"] = int(np.floor(config_dict["SIMULATION_FREQ_HZ"]/config_dict["CONTROL_FREQ_HZ"])) # TODO: @Andrew 
        config_dict["OUTPUT_FOLDER"] = os.path.join(trajectory.root, "sim_data")
        config_dict["T_HORIZON"] = 1.1*trajectory.T_horizon # TODO: @Andrew
    if config_dict["GUI"]:
        try:
            config_dict["VIZ"] = get_viz()
        except:
            pass
    
    config = namedtuple("tracking_config", config_dict.keys())(**config_dict)
    return config


######################## PYBULLET VISUALIZATION ########################
########################################################################


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
    if obs is not None: # Render smaller spheres for Trajectory and Flight Path
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


######################## MPL VIZUALIZATION ########################
###################################################################


def plot_trajectories_by_task(
        task_group_trajectory_dict, 
        root=DEFAULT_ROOT, 
        dataset_name=DEFAULT_DATASET_NAME, 
        task_group=DEFAULT_TASK_NAME
    ):
    """
    Plot waypoints and trajectories from a given group of tasks in a task_battery
    Saves plots to the task_plots/ subdirectory of the dataset_name/ directory

    Parameters:
        - task_group_trajectory_dict: dict() - dictionary of (task_name, task_trajectory_dict) key-value pairs
        - root: str - The root path for where datasets are located
        - dataset_name: str - The name of the dataset
        - task_name: str - The name of the current task

    """
    waypoints = task_group_trajectory_dict[task_group]["waypoints"]
    x_traj = task_group_trajectory_dict[task_group]["x"]

    plt_root = os.path.join(root, dataset_name, "task_plots")
    if not os.path.exists(plt_root):
        os.mkdir(plt_root)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection="3d")
    # WAYPOINTS
    dense_wpts, sparse_wpts = waypoints["dense"], waypoints["sparse"]
    n_paths = len(dense_wpts)
    step_factor = 10
    for i in range(0, n_paths, step_factor):
        dense_path = dense_wpts[i].T
        sparse_path = sparse_wpts[i].T
        # dense_path[0, :] += int(i / (2*step_factor))
        # sparse_path[0, :] += int(i / (2*step_factor))
        ax.plot(*dense_path, color="k", label="dense path", linewidth=1)
        ax.scatter(*sparse_path, s=25, label="sparse waypoints", marker="o")
    fig.suptitle(f"{task_group} waypoints")
    plt.savefig(os.path.join(plt_root, f"{task_group}_wpts.png"))

    fig2 = plt.figure(figsize=(12, 7))
    ax2 = fig2.add_subplot(projection="3d")
    # TRAJECTORIES
    for i in range(0, n_paths, step_factor):
        traj = x_traj[i].T
        traj = traj[:, np.sum(traj, axis=0) != -300.0].reshape((3, -1))
        # traj[0, :] += int(i / (2*step_factor))
        p = ax2.plot(*traj, label="trajectory")
    fig2.suptitle(f"{task_group} trajectories")
    plt.savefig(os.path.join(plt_root, f"{task_group}_traj.png"))

    # plt.show()
    plt.close("all")