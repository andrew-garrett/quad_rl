#################### IMPORTS ####################
#################################################


import os
import sys
# setting path
sys.path.append("../quad_rl")
import argparse
import threading
import json

from bootstrap.task_battery import TaskBattery
from bootstrap.task_gen import Tasks
from bootstrap.utils import cleanup
from traj_gen import yield_all_task_trajectories
import traj_track


#################### GLOBAL VARIABLES ####################
##########################################################


# Get the physics model from ./configs/tracking/tracking_config.json and augment the ROOT to account for different physics models
CONFIG_FPATHS = ["./configs/tracking/default_tracking_config.json"] # For barebones visualization --------------------------------------------------- Produces zip, zip matches expectation
CONFIG_FPATHS.append("./configs/tracking/debug_tracking_config.json") # For debug visualization ----------------------------------------------------- Produces zip, zip matches expectation
CONFIG_FPATHS.append("./configs/tracking/data_tracking_config.json") # For data-collection ---------------------------------------------------------- Produces zip, zip matches expectation
CONFIG_FPATHS.append("./configs/tracking/video_data_tracking_config.json") # For data-collection with video (barebones visualization) --------------- Produces zip, need to check
CONFIG_FPATHS.append("./configs/tracking/debug_video_data_tracking_config.json") # For data-collection with video (debug visualization) ------------- Produces zip, zip matches expectation


VERBOSE = False


#################### GENERATE TRAJECTORIES FOR TASK BATTERY ####################
################################################################################


def collect_bootstrap_data(
        root,
        task_battery,
        config_fpath="./configs/tracking/tracking_config.json"
    ):
    """
    Function to create an initial bootstrapped training dataset

        1. Generate waypoint csv files for each task case (parameterized by path-planning parameters in dataset_name_TASK_CONFIG.json)
        2. For each waypoint csv file, plan several trajectories (parameterized by trajectory-planning parameters in dataset_name_TASK_CONFIG.json)
        3. For each trajectory, collect simulation data (parameterized by simulation and trajectory-tracking parameters in dataset_name_TRACKING_CONFIG.json)
    """
    # Generate waypoint datasets
    dataset_name = Tasks.generate_tasks(root=root, task_battery=task_battery)
    
    # Generate Task Trajectories
    trajectories_by_task = {}
    prev_task_group = None
    # iterate through all trajectories
    for i, (_, task, traj_gen_obj) in enumerate(yield_all_task_trajectories(
        root=root, 
        dataset_name=dataset_name, 
        verbose=VERBOSE
    )):
        task_group = "_".join(task.split("_")[1:3])
        # if we have reached a new group of tasks
        if task_group not in trajectories_by_task.keys():
            if prev_task_group is not None:
                # for each trajectory in the previous group of tasks, collect desired states
                for traj in trajectories_by_task[prev_task_group]["generated_trajectories"]:
                    traj_track.track(traj, config_fpath=config_fpath, verbose=VERBOSE)

            # then, initialize the next group of tasks
            trajectories_by_task[task_group] = {
                "generated_trajectories": []
            }
        prev_task_group = task_group
        trajectories_by_task[task_group]["generated_trajectories"].append(traj_gen_obj)
    
    for traj in trajectories_by_task[task_group]["generated_trajectories"]:
        traj_track.track(traj, config_fpath=config_fpath, verbose=VERBOSE)

    cleanup(root, dataset_name, config_fpath=config_fpath)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap Simulation Dataset Collection Script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t_battery_choices = [TASK_BATTERY.name for _, TASK_BATTERY in enumerate(TaskBattery)]
    t_battery_choices.append("FULL")
    parser.add_argument("--task-battery", default="DEBUG", type=str, help="task_battery.TaskBattery", choices=t_battery_choices)
    tracking_options = ["_".join(t_config_fpath.split("/")[-1].split("_")[:-2]) for t_config_fpath in CONFIG_FPATHS]
    parser.add_argument("--tracking-config", default="default", type=str, help="controls what kind of visualization and whether or not we collect data", choices=tracking_options)
    ARGS = parser.parse_args()

    ##### Get the main/base tracking_config.json and update relevant values from the argument 
    base_config_fpath = "./configs/tracking/tracking_config.json"
    with open(base_config_fpath, "r") as f:
        base_config = json.load(f)
        ROOT = os.path.join("./bootstrap/datasets/", base_config["PHYSICS"])
    # Get the visualization/data collection config specified by the CLI argument 
    config_fpath = f"./configs/tracking/{ARGS.tracking_config}_tracking_config.json"
    # config_fpath = f"./configs/tracking/data_tracking_config.json"
    if "debug" in config_fpath:
        VERBOSE = True
    with open(config_fpath, "r") as f:
        config = json.load(f)
        for k, v in config.items():
            base_config[k] = v
    # Save the updated config for later copying it into the appropriate dataset
    with open(base_config_fpath, "w") as f:
        json.dump(base_config, f, indent="\t", sort_keys=True)

    ##### Choose the task battery to run simulations on
    if ARGS.task_battery != "FULL":
        for _, t_battery in enumerate(TaskBattery):
            if t_battery.name == ARGS.task_battery:
                collect_bootstrap_data(ROOT, t_battery)
    else:
        # If we are performing multithreaded dataset collection, revert to the most lightweight tracking config
        config_fpath = "./configs/tracking/data_tracking_config.json"
        with open(config_fpath, "r") as f:
            config = json.load(f)
            for k, v in config.items():
                base_config[k] = v
        # Save the updated config for later copying it into the appropriate dataset
        with open(base_config_fpath, "w") as f:
            json.dump(base_config, f, indent="\t", sort_keys=True)
        
        ##### Start Threads, each collecting a dataset
        threads = []
        for _, t_battery in enumerate(TaskBattery):
            x = threading.Thread(target=collect_bootstrap_data, args=(ROOT, t_battery))
            threads.append(x)
            x.start()
        
        for thread in threads:
            thread.join()