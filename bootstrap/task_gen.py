#################### IMPORTS ####################
#################################################


from enum import Enum
import os
import json
import csv
from itertools import product
import numpy as np

from task_battery import *


#################### GLOBAL PARAMETERS ####################
###########################################################


DEFAULT_DATASET_NAME = f"{DEFAULT_TASK_BATTERY.name}000"


#################### GENERATE WAYPOINTS FOR TASK BATTERY ####################
#############################################################################


class Tasks:
    """
    General Class to keep track of how each task is generated
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_takeoff_tasks(params, root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME):
        param_grid = get_param_grid(params)
        for i, param_grid_i in enumerate(param_grid):
            continue
        print("Takeoff Tasks Generated")
        return

    @staticmethod
    def generate_landing_tasks(params, root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME):
        param_grid = get_param_grid(params)
        for i, param_grid_i in enumerate(param_grid):
            continue
        print("Landing Tasks Generated")
        return

    @staticmethod
    def generate_hover_tasks(params, root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME):
        param_grid = get_param_grid(params)
        for i, param_grid_i in enumerate(param_grid):
            continue
        print("Hover Tasks Generated")
        return

    @staticmethod
    def generate_linear_step_tasks(params, root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME):
        """
        For x,y,z directions, set the initial position to be a unit step away from the origin
        with orientation as the identity.  Generate a single waypoint at the origin.
        """
        param_grid = get_param_grid(params)
        for i, param_grid_i in enumerate(param_grid):
            ax = param_grid_i[0]
            task_fname = f"{root}{dataset_name}/waypoints/task_linear_step_ax-{ax}.csv"
            if os.path.exists(task_fname):
                continue
            else:
                with open(task_fname, "w", newline="") as f:
                    csv_writer = csv.writer(f)
                    header = ["x", "y", "z"]
                    if ax == "x":
                        row = [1.0, 0.0, 0.0]
                    elif ax == "y":
                        row = [0.0, 1.0, 0.0]
                    else:
                        row = [0.0, 0.0, 1.0]
                    footer = [0.0, 0.0, 0.0]

                    csv_writer.writerow(header)
                    csv_writer.writerow(row)
                    csv_writer.writerow(footer)
        print("Linear-Step Tasks Generated")
        return

    @staticmethod
    def generate_angular_step_tasks(params, root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME):
        """
        For x,y,z directions, set the initial orientation to be a step away from the identity orientation.
        Generate a single waypoint at the origin.
        """
        param_grid = get_param_grid(params)
        for i, param_grid_i in enumerate(param_grid):
            ax = param_grid_i[0]
            task_fname = f"{root}{dataset_name}/waypoints/task_angular_step_ax-{ax}.csv"
            if os.path.exists(task_fname):
                continue
            else:
                with open(task_fname, "w", newline="") as f:
                    csv_writer = csv.writer(f)
                    header = ["x", "y", "z"]
                    if ax == "x":
                        footer = [0.0, 0.0, 0.0]
                    elif ax == "y":
                        footer = [0.0, 0.0, 0.0]
                    else:
                        footer = [0.0, 0.0, 0.0]

                    csv_writer.writerow(header)
                    csv_writer.writerow(footer)
        print("Angular-Step Tasks Generated")
        return

    @staticmethod
    def generate_straight_away_tasks(params, root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME):
        """
        For x,y,z directions, start at 5m away from the origin with orientation as the identity.
        Generate a single waypoint at the origin.
        """
        param_grid = get_param_grid(params)
        for i, param_grid_i in enumerate(param_grid):
            ax = param_grid_i[0]
            task_fname = f"{root}{dataset_name}/waypoints/task_straight_away_ax-{ax}.csv"
            if os.path.exists(task_fname):
                continue
            else:
                with open(task_fname, "w", newline="") as f:
                    csv_writer = csv.writer(f)
                    header = ["x", "y", "z"]
                    if ax == "x":
                        row = [5.0, 0.0, 0.0]
                    elif ax == "y":
                        row = [0.0, 5.0, 0.0]
                    else:
                        row = [0.0, 0.0, 5.0]
                    footer = [0.0, 0.0, 0.0]
                    
                    csv_writer.writerow(header)
                    csv_writer.writerow(row)
                    csv_writer.writerow(footer)
        print("Straight-Away Tasks Generated")
        return

    @staticmethod
    def generate_figure_eight_tasks(params, root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME):
        """
        For x,y,z directions, generate waypoints in figure eight patterns
        """
        param_grid = get_param_grid(params)
        for i, param_grid_i in enumerate(param_grid):
            ax, dh, r, res = param_grid_i
            task_fname = f"{root}{dataset_name}/waypoints/task_figure_eight_ax-{ax}_radii-{r}_dh-{dh}_res-{res}.csv"
            if os.path.exists(task_fname):
                continue
            else:
                with open(task_fname, "w", newline="") as f:
                    csv_writer = csv.writer(f)
                    header = ["x", "y", "z"]
                    csv_writer.writerow(header)

                    num_wpts = int(DEFAULT_T / res**2)
                    t = np.linspace(0, DEFAULT_T, num=num_wpts)
                    long_axis = np.cos(r*t)
                    short_axis = 0.5 * np.sin(r*t) * long_axis
                    perp_axis = np.sin(dh*t)
                    for t_step in range(num_wpts):
                        if ax == 'x':
                            row = [long_axis[t_step], short_axis[t_step], perp_axis[t_step]]
                        elif ax == 'y':
                            row = [perp_axis[t_step], long_axis[t_step], short_axis[t_step]]
                        else:
                            row = [short_axis[t_step], perp_axis[t_step], long_axis[t_step]]
                        csv_writer.writerow(row)
        print("Figure-Eight Tasks Generated")
        return

    @staticmethod
    def generate_tasks(root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME, task_battery=DEFAULT_TASK_BATTERY):
        # Make the dataset directory
        if os.path.exists(f"{root}{dataset_name}/waypoints/"):
            counter = 0
            while True:
                if os.path.exists(f"{root}dataset{str(counter).zfill(3)}/waypoints/"):
                    counter += 1
                else:
                    dataset_name = f"dataset{str(counter).zfill(3)}"
                    break
        os.makedirs(f"{root}{dataset_name}/waypoints/")

        # Iterate through the tasking battery and generate waypoint files
        battery = task_battery.value
        for key, value in battery.items():
            # generate task by calling relevant function
            getattr(Tasks, value["taskcase_generator"])(value['params'], root, dataset_name)
            battery[key] = value["params"]
        
        # Save the dataset parameters
        with open(f"{root}{dataset_name}/{dataset_name}.json", "w") as f:
            json.dump(battery, f, indent="\t")


######################## UTILITY FUNCTION ########################
##################################################################


def get_param_grid(params):
    """
    Get the grid of parameters we'd like to create taskcases for
    """
    param_spaces = [v for k, v in sorted(params.items(), key=lambda x: x[0]) if k not in TRAJECTORY_PARAMS]
    return product(*param_spaces)


######################## RUNNER ########################
########################################################


if __name__ == "__main__":

    Tasks.generate_tasks()