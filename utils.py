#################### IMPORTS ####################
#################################################


import os
import subprocess
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


def cleanup(root, dataset_name, config_fpath="./configs/tracking/tracking_config.json"):
    """
    Zip a dataset and delete the uncompressed folder
    Also has functionality to save mp4's of flight trials from 
    """
    data_root = os.path.join(root, dataset_name)
    video_output_folder = os.path.join(data_root, 'videos')
    ##### Copy the tracking config to the dataset root
    shutil.copy(config_fpath, data_root)
    os.rename(os.path.join(data_root, "tracking_config.json"), os.path.join(data_root, f"{dataset_name}_TRACKING_CONFIG.json"))
    with open(os.path.join(data_root, f"{dataset_name}_TRACKING_CONFIG.json"), "r") as f:
        video_fps = json.load(f)["CONTROL_FREQ_HZ"] // 2
    ##### Walk the dataset directory and zip it's contents
    with ZipFile(f"{data_root}.zip", "w") as zip_object:
        for folder, sub_folders, f_names in os.walk(data_root):
            png_fnames = []
            zipped_fnames =  []
            for f_name in f_names:
                f_path = os.path.join(folder, f_name)
                # Only make one video per folder of images
                if ".png" in f_name and f_path not in png_fnames and "frame" in f_name:
                    ##### If we have pngs saved, create videos with ffmpeg
                    if not os.path.exists(video_output_folder): 
                        os.makedirs(video_output_folder, exist_ok=True)
                    command = f"ffmpeg -y -framerate {video_fps} -s 960x540 -i {os.path.join(folder, 'frame_%d.png')} -c:v libx264 -pix_fmt yuv420p {os.path.join(video_output_folder, str(os.path.basename(folder)).replace('.', '-') + '.mp4')}".split(" ")
                    if subprocess.run(command).returncode == 0:
                        print ("FFmpeg Script Ran Successfully")
                    else:
                        print ("There was an error running your FFmpeg script")
                        exit()
                    ##### Dont save the pngs
                    png_fnames = [os.path.join(folder, fn) for fn in f_names if ".png" in fn]
                if f_path not in png_fnames and f_path not in zipped_fnames:
                    zip_object.write(f_path, f_path[len(data_root):])
                    zipped_fnames.append(f_path)
        if os.path.exists(video_output_folder):
            for video_fname in os.listdir(video_output_folder):
                video_fpath = os.path.join(video_output_folder, video_fname)
                zip_object.write(video_fpath, video_fpath[len(data_root):])
    if os.path.exists(f"{data_root}.zip"):
        ##### Delete the uncompressed dataset folder
        # shutil.rmtree(data_root)
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


def get_tracking_config(trajectory=None, config_fpath="./configs/tracking/tracking_config.json"):
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
        config_dict["OUTPUT_FOLDER"] =  os.path.join(trajectory.root, "sim_data")
        config_dict["T_FINISH"] = 1.1*trajectory.t_finish # TODO: @Andrew
    if config_dict["GUI"] or config_dict["USER_DEBUG_GUI"]:
        try:
            config_dict["VIZ"] = get_viz(config_dict)
        except:
            pass
    
    config = namedtuple("tracking_config", config_dict.keys())(**config_dict)
    return config


######################## PYBULLET VISUALIZATION ########################
########################################################################


def get_viz(config_dict):
    """
    Creates a config object for visualization parameters
    """
    sphereRadius = 0.05
    markers = {}
    markers["waypointSphereId"] = p.createVisualShape(
        shapeType=p.GEOM_SPHERE, 
        rgbaColor=[1, 0, 0, 1],
        radius=sphereRadius*0.4
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

def render_rollouts(config, ref_traj_arr, rollout_traj_arr):
    """
    Render the reference trajectory and the mppi rollout trajectory
    """
    viz_config_dict = config.VIZ._asdict()
    if "refTrajLineIds" not in viz_config_dict.keys() or "rolloutTrajLineIds" not in viz_config_dict.keys():
        viz_config_dict["refTrajLineIds"] = []
        viz_config_dict["rolloutTrajLineIds"] = []
        for ref_traj in ref_traj_arr:
            ref_speeds = np.linalg.norm(ref_traj[:, 3:6], axis=1)
            ref_speeds_norm = np.linalg.norm(ref_speeds)
            if ref_speeds_norm > 0:
                ref_speeds /= ref_speeds_norm
            viz_config_dict["refTrajLineIds"].extend([p.addUserDebugLine(ref_traj[t, :3], ref_traj[t+1, :3], [0., 0., ref_speeds[t]], lineWidth=5) for t in range(ref_traj.shape[0]-1)])
        for rollout_traj in rollout_traj_arr:
            rollout_speeds = np.linalg.norm(rollout_traj[:, 3:6], axis=1)
            rollout_speeds_norm = np.linalg.norm(rollout_speeds)
            if rollout_speeds_norm > 0:
                rollout_speeds /= rollout_speeds_norm
            viz_config_dict["rolloutTrajLineIds"].extend([p.addUserDebugLine(rollout_traj[t, :3], rollout_traj[t+1, :3], [0., rollout_speeds[t], 0.], lineWidth=5) for t in range(rollout_traj.shape[0]-1)])
        viz_config = namedtuple("viz_config", viz_config_dict.keys())(**viz_config_dict)
        config_dict = config._asdict()
        config_dict["VIZ"] = viz_config
        config = namedtuple("tracking_config", config_dict.keys())(**config_dict)
    else:
        for ref_traj in ref_traj_arr:
            ref_speeds = np.linalg.norm(ref_traj[:, 3:6], axis=1)
            ref_speeds_norm = np.linalg.norm(ref_speeds)
            if ref_speeds_norm > 0:
                ref_speeds /= ref_speeds_norm
            for t in range(ref_traj.shape[0] - 1):
                p.addUserDebugLine(ref_traj[t, :3], ref_traj[t+1, :3], [0., 0., ref_speeds[t]], lineWidth=5, replaceItemUniqueId=config.VIZ.refTrajLineIds[t])
        for rollout_traj in rollout_traj_arr:
            rollout_speeds = np.linalg.norm(rollout_traj[:, 3:6], axis=1)
            rollout_speeds_norm = np.linalg.norm(rollout_speeds)
            if rollout_speeds_norm > 0:
                rollout_speeds /= rollout_speeds_norm
            for t in range(rollout_traj.shape[0] - 1):
                p.addUserDebugLine(rollout_traj[t, :3], rollout_traj[t+1, :3], [0., rollout_speeds[t], 0.], lineWidth=5, replaceItemUniqueId=config.VIZ.rolloutTrajLineIds[t])
    return config



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