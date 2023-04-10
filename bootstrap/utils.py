#################### IMPORTS ####################
#################################################


import os
import json
import numpy as np
import matplotlib.pyplot as plt

from task_gen import TRAJECTORY_PARAMS, DEFAULT_T, DEFAULT_DATASET_NAME, DEFAULT_ROOT


#################### GLOBAL VARIABLES ####################
##########################################################


DEFAULT_TASK_NAME = "linear_step.csv"


######################## UTILITY FUNCTIONS ########################
###################################################################


def collect_task_trajectory(task_trajectories, traj):
    """
    Function for running one trial of simulating a trajectory
    """
    T = traj.t_start_vec.flatten()[-1]
    t_vector = np.linspace(0, T, num=1000)
    warmdown_iters = 30
    # tracking waypoints
    if "waypoints" not in task_trajectories.keys():
        task_trajectories["waypoints"] = {
            "sparse": [traj.points],
            "dense": [traj.path]
        }
    else: 
        task_trajectories["waypoints"]["sparse"].append(traj.points)
        task_trajectories["waypoints"]["dense"].append(traj.path)

    # simulate the trajectory for T timesteps
    x_traj = -100.0 * np.ones((1, t_vector.shape[0], 3))                                                                                                                             
    x_dot_traj = -100.0 * np.ones((1, t_vector.shape[0], 3))
    for j, t in enumerate(t_vector):
        # for each timestep, record the state
        for k, v in traj.update(t).items():
            if k == "x":
                x_traj[:, j, :] = v
            elif k == "x_dot":
                x_dot_traj[:, j, :] = v
        if traj.is_done:
            # run 30 extra timesteps to allow system to stabilize/warmdown
            warmdown_iters -= 1
            if warmdown_iters == 0:
                break
    if "x" not in task_trajectories.keys():
        task_trajectories["x"] = x_traj
        task_trajectories["x_dot"] = x_dot_traj
    else:
        task_trajectories["x"] = np.vstack((task_trajectories["x"], x_traj))
        task_trajectories["x_dot"] = np.vstack((task_trajectories["x_dot"], x_dot_traj))
    
    return task_trajectories


def get_task_params(root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME, task_name=DEFAULT_TASK_NAME):
    """
    Prepare parameters for planning a trajectory
    """

    # load params
    with open(f"{root}{dataset_name}/{dataset_name}.json", "r") as f:
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


def plot_trajectories_by_task(task_group_trajectory_dict, root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME, task_group=DEFAULT_TASK_NAME):
    """
    Function to plot waypoints and trajectories from a given group of tasks
    """
    waypoints = task_group_trajectory_dict[task_group]["waypoints"]
    x_traj = task_group_trajectory_dict[task_group]["x"]
    x_dot_traj = task_group_trajectory_dict[task_group]["x_dot"]
    speed_traj = np.linalg.norm(x_dot_traj, axis=(-1))

    plt_root = f"{root}{dataset_name}/task_plots/"
    if not os.path.exists(plt_root):
        os.mkdir(plt_root)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(projection='3d')
    # WAYPOINTS
    dense_wpts, sparse_wpts = waypoints["dense"], waypoints["sparse"]
    n_paths = len(dense_wpts)
    step_factor = 20
    for i in range(0, n_paths, step_factor):
        dense_path = dense_wpts[i].T
        sparse_path = sparse_wpts[i].T
        # dense_path[0, :] += int(i / (2*step_factor))
        # sparse_path[0, :] += int(i / (2*step_factor))
        ax.plot(*dense_path, color="k", label="dense path", linewidth=1)
        ax.scatter(*sparse_path, s=25, label="sparse waypoints", marker="o")
    fig.suptitle(f"{task_group} waypoints")
    plt.savefig(f"{plt_root}{task_group}_wpts.png")

    fig2 = plt.figure(figsize=(12, 7))
    ax2 = fig2.add_subplot(projection='3d')
    # TRAJECTORIES
    for i in range(0, n_paths, step_factor):
        traj = x_traj[i].T

        traj = traj[:, np.sum(traj, axis=0) != -300.0].reshape((3, -1))
        # traj[0, :] += int(i / (2*step_factor))
        p = ax2.plot(*traj, label="trajectory")
    # fig2.colorbar(p)
    fig2.suptitle(f"{task_group} trajectories")
    plt.savefig(f"{plt_root}{task_group}_traj.png")

    plt.show()
    plt.close("all")