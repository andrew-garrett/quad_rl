#################### IMPORTS ####################
#################################################


from tqdm import tqdm
from itertools import product
import numpy as np

import task_battery
from task_gen import generate_tasks
from traj_gen import yield_all_task_trajectories


#################### GLOBAL VARIABLES ####################
##########################################################

ROOT = "./bootstrap/datasets/"

DATASET_NAME = "dataset000"


#################### GENERATE TRAJECTORIES FOR TASK BATTERY ####################
################################################################################


def collect_bootstrap_data(root, dataset_name, num_iterations, num_trials):
    """
    Function to create an initial bootstrapped training dataset

        1. Generate waypoint csv files for each task_case (parameterized by path-planning parameters)
        2. For each waypoint csv, plan several trajectories according (parameterized by trajectory-planning parameters)
        3. For each trajectory, collect simulation data from num_iterations of num_trials of the following that trajectory
            X is all (state_i, control_i) pairs (70% of each trajectory's mini-dataset)
            y is all state_{i+1} vectors
    """
    # Generate waypoint csv's
    generate_tasks(root, dataset_name)

    # Generate trajectories for each waypoint csv
    trajectories = yield_all_task_trajectories(root, dataset_name)
    for i, traj_i in enumerate(trajectories):
        continue