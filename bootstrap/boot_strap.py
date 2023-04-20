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


# Get the physics model from the tracking_config.json and augment the ROOT to account for different physics models
with open("./configs/tracking_config.json", "r") as f:
    ROOT = os.path.join("./bootstrap/datasets/", json.load(f)["PHYSICS"])

VERBOSE = False


#################### GENERATE TRAJECTORIES FOR TASK BATTERY ####################
################################################################################


def collect_bootstrap_data(
        root,
        task_battery
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
                    traj_track.track(traj, verbose=VERBOSE)

            # then, initialize the next group of tasks
            trajectories_by_task[task_group] = {
                "generated_trajectories": []
            }
        prev_task_group = task_group
        trajectories_by_task[task_group]["generated_trajectories"].append(traj_gen_obj)
    
    for traj in trajectories_by_task[task_group]["generated_trajectories"]:
        traj_track.track(traj, verbose=VERBOSE)

    cleanup(root, dataset_name)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Bootstrap Simulation Dataset Collection Script")
    t_battery_choices = [TASK_BATTERY.name for _, TASK_BATTERY in enumerate(TaskBattery)]
    t_battery_choices.append("FULL")
    parser.add_argument("--task-battery", default="DEBUG", type=str, help="task_battery.TaskBattery", metavar="", choices=t_battery_choices)
    ARGS = parser.parse_args()
    if ARGS.task_battery != "FULL":
        for _, t_battery in enumerate(TaskBattery):
            if t_battery.name == ARGS.task_battery:
                collect_bootstrap_data(ROOT, t_battery)
    else:
        threads = []
        for _, t_battery in enumerate(TaskBattery):
            x = threading.Thread(target=collect_bootstrap_data, args=(ROOT, t_battery))
            threads.append(x)
            x.start()
        
        for thread in threads:
            thread.join()