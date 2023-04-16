#################### IMPORTS ####################
#################################################


import sys
# setting path
sys.path.append('..\\quad_rl')
import argparse
import threading
import bootstrap.task_battery as task_battery
from bootstrap.task_gen import Tasks
from bootstrap.utils import cleanup
from traj_gen import *
import traj_track


#################### GLOBAL VARIABLES ####################
##########################################################

ROOT = "./bootstrap/datasets/"

VERBOSE = False


#################### GENERATE TRAJECTORIES FOR TASK BATTERY ####################
################################################################################


def collect_bootstrap_data(
        TASK_BATTERY
    ):
    """
    Function to create an initial bootstrapped training dataset

        1. Generate waypoint csv files for each task_case (parameterized by path-planning parameters)
        2. For each waypoint csv, plan several trajectories according (parameterized by trajectory-planning parameters)
        3. For each trajectory, collect simulation data from num_iterations of num_trials of the following that trajectory
            X is all (state_i, control_i) pairs (70% of each trajectory's mini-dataset)
            y is all state_{i+1} vectors
    """
    # Generate waypoint datasets
    DATASET_NAME = f"{TASK_BATTERY.name}_000"
    DATASET_NAME = Tasks.generate_tasks(root=ROOT, dataset_name=DATASET_NAME, task_battery=TASK_BATTERY)
    
    # Generate Task Trajectories
    trajectories_by_task = {}
    prev_task_group = None
    # iterate through all trajectories
    for i, (_, task, traj_gen_obj) in enumerate(yield_all_task_trajectories(
        root=ROOT, 
        dataset_name=DATASET_NAME, 
        verbose=VERBOSE
    )):
        task_group = "_".join(task.split("_")[1:3])
        # if we have reached a new group of tasks
        if task_group not in trajectories_by_task.keys():
            if prev_task_group is not None:
                # for each trajectory in the previous group of tasks, collect desired states
                for traj in trajectories_by_task[prev_task_group]["generated_trajectories"]:
                    traj_track.track(traj)
            
            # then, initialize the next group of tasks
            trajectories_by_task[task_group] = {
                "generated_trajectories": []
            }
        prev_task_group = task_group
        trajectories_by_task[task_group]["generated_trajectories"].append(traj_gen_obj)
    
    for traj in trajectories_by_task[task_group]["generated_trajectories"]:
        traj_track.track(traj)

    cleanup(ROOT, DATASET_NAME)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trajectory Tracking Script')
    t_battery_choices = [TASK_BATTERY.name for _, TASK_BATTERY in enumerate(task_battery.TaskBattery)]
    t_battery_choices.append("FULL")
    parser.add_argument('--task-battery', default="DEBUG", type=str, help='task_battery.TaskBattery', metavar='', choices=t_battery_choices)
    ARGS = parser.parse_args()
    if ARGS.task_battery != "FULL":
        for _, t_battery in enumerate(task_battery.TaskBattery):
            if t_battery.name == ARGS.task_battery:
                collect_bootstrap_data(t_battery)
    else:
        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     executor.map(collect_bootstrap_data, [t_battery for _, t_battery in enumerate(task_battery.TaskBattery)])
        threads = []
        for _, t_battery in enumerate(task_battery.TaskBattery):
            x = threading.Thread(target=collect_bootstrap_data, args=(t_battery,))
            threads.append(x)
            x.start()
        
        for thread in threads:
            thread.join()