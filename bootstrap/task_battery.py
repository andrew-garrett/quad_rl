#################### IMPORTS ####################
#################################################


from enum import Enum
import numpy as np


#################### GLOBAL VARIABLES ####################
##########################################################


NO_SPEED = 0.0 # m/s
BASE_SPEED = 1.0 # m/s
LOW_SPEED = 0.5*BASE_SPEED # m/s
MED_SPEED = 1.0*BASE_SPEED # m/s
HIGH_SPEED = 1.5*BASE_SPEED # m/s
AGGRO_MULTIPLIER = 1.5
NUM_DRONES = 1 # number of drones along a single axis of a 3D cube configuration

DEFAULT_ROOT = "./bootstrap/datasets/pyb/"
DEFAULT_TASK_NAME = "linear_step.csv"
TRAJECTORY_PARAMS = ("num_drones", "speed", "rdp_threshold", "trajectory_generator")
DEFAULT_T = 2*np.pi


#################### FULL BATTERY ######################
########################################################


FULL_TASK_BATTERY = {

    "takeoff": {
        "taskcase_generator": "generate_takeoff_tasks",
        "params": {
            "num_drones": 1,
            "speed": [LOW_SPEED, HIGH_SPEED]
        }
    },
    "landing": {
        "taskcase_generator": "generate_landing_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "speed": [LOW_SPEED, MED_SPEED]
        }
    },
    "hover": {
        "taskcase_generator": "generate_hover_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "speed": [NO_SPEED]
        }
    },
    "linear_step": {
        "taskcase_generator": "generate_linear_step_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "y", "z"],
            "speed": [LOW_SPEED, MED_SPEED, HIGH_SPEED]
        }
    },
    "angular_step": {
        "taskcase_generator": "generate_angular_step_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["r", "p", "y"],
            "rpy0": [5.0, 10.0, 20.0, 30.0, 45.0, 60.0],
            "speed": [LOW_SPEED]
        }
    },
    "straight_away": {
        "taskcase_generator": "generate_straight_away_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "y", "z"],
            "speed": [MED_SPEED, HIGH_SPEED]
        }
    },
    "figure_eight": {
        "taskcase_generator": "generate_figure_eight_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "y", "z"],
            "dh": [0.0, 1.0, 2.0],
            "radii": [0.25, 1.0, 2.0],
            "rdp_threshold": [0.025, 0.1, 0.25],
            "res": [0.01, 0.05, 0.2],
            "speed": [LOW_SPEED, MED_SPEED, HIGH_SPEED],
            "trajectory_generator": [
                # "constant_speed", 
                # "cubic_spline", 
                "min_snap"
            ]
        }
    }
}


#################### CONTROLLED TASK BATTERY ####################
#################################################################


CONTROLLED_TASK_BATTERY = {

    "straight_away": {
        "taskcase_generator": "generate_straight_away_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "z"],
            "speed": [LOW_SPEED, MED_SPEED]
        }
    },
    "figure_eight": {
        "taskcase_generator": "generate_figure_eight_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "z"],
            "dh": [0.0, 1.0],
            "radii": [1.0, 2.0],
            "rdp_threshold": [0.025, 0.1, 0.25],
            "res": [0.05, 0.2],
            "speed": [LOW_SPEED, MED_SPEED],
            "trajectory_generator": [
                # "constant_speed", 
                # "cubic_spline", 
                "min_snap"
            ]
        }
    }
}


#################### AGGRESSIVE TASK BATTERY ####################
#################################################################


AGGRESSIVE_TASK_BATTERY = {

    "straight_away": {
        "taskcase_generator": "generate_straight_away_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["y", "z"],
            "speed": [AGGRO_MULTIPLIER*MED_SPEED, AGGRO_MULTIPLIER*HIGH_SPEED]
        }
    },
    "figure_eight": {
        "taskcase_generator": "generate_figure_eight_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["y", "z"],
            "dh": [0.0, 2.0],
            "radii": [1.0, 4.0],
            "rdp_threshold": [0.025, 0.1, 0.25],
            "res": [0.05, 0.2],
            "speed": [AGGRO_MULTIPLIER*MED_SPEED, AGGRO_MULTIPLIER*HIGH_SPEED],
            "trajectory_generator": [
                # "constant_speed", 
                # "cubic_spline", 
                "min_snap"
            ]
        }
    }
}


#################### DEBUG TASK BATTERY ####################
############################################################


DEBUG_TASK_BATTERY = {

    "straight_away": {
        "taskcase_generator": "generate_straight_away_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "y"],
            "speed": [MED_SPEED, HIGH_SPEED]
        }
    },
    "figure_eight": {
        "taskcase_generator": "generate_figure_eight_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x"],
            "dh": [0.0],
            "radii": [1.5, 3.0],
            "rdp_threshold": [0.025, 0.1, 0.25],
            "res": [0.25],
            "speed": [HIGH_SPEED],
            "trajectory_generator": [
                # "constant_speed",
                # "cubic_spline", 
                "min_snap"
            ]
        }
    }
}


#################### TEST TASK BATTERY ####################
###########################################################


TEST_TASK_BATTERY = {
    "racetrack": {
        "taskcase_generator": "generate_racetrack_tasks",
        "params": {
            "num_drones": 1,
            "ax": ["x"],
            "dh": [0.0, 3.0],
            "radii": [2.0, 6.0],
            "rdp_threshold": [0.025, 0.1, 0.25],
            "res": [0.25, 0.5],
            "speed": [HIGH_SPEED, AGGRO_MULTIPLIER*HIGH_SPEED],
            "trajectory_generator": [
                "min_snap"
            ]
        }
    },
    "avoid": {
        "taskcase_generator": "generate_avoid_tasks",
        "params": {
            "num_drones": 1,
            "ax": ["x"],
            "dh": [0.0],
            "radii": [1.0, 2.0],
            "rdp_threshold": [0.025, 0.1, 0.25],
            "res": [0.25, 0.5],
            "speed": [MED_SPEED],
            "trajectory_generator": [
                "min_snap"
            ]
        }
    }
}


#################### TASK BATTERY ENUM ####################
###########################################################


class TaskBattery(Enum):

    AGGRO = AGGRESSIVE_TASK_BATTERY

    CONTROLLED = CONTROLLED_TASK_BATTERY

    DEBUG = DEBUG_TASK_BATTERY

    # FULL = FULL_TASK_BATTERY

    # TEST = TEST_TASK_BATTERY

DEFAULT_TASK_BATTERY = TaskBattery.DEBUG
DEFAULT_DATASET_NAME = f"{DEFAULT_TASK_BATTERY.name}_000"