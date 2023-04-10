#################### IMPORTS ####################
#################################################


from enum import Enum
import numpy as np


#################### GLOBAL VARIABLES ####################
##########################################################


NO_SPEED = 0.0 # m/s
LOW_SPEED = 1.0 # m/s
MED_SPEED = 3.0 # m/s
HIGH_SPEED = 6.0 # m/s
NUM_DRONES = 1

DEFAULT_ROOT = "./bootstrap/datasets/"
TRAJECTORY_PARAMS = ("num_drones", "speed", "rdp_threshold", "trajectory_generator")
DEFAULT_T = 5*np.pi


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
                "constant_speed", 
                "cubic_spline", 
                "min_snap"
            ]
        }
    }
}


#################### SMALL TASK BATTERY ####################
############################################################


SMALL_TASK_BATTERY = {

    "linear_step": {
        "taskcase_generator": "generate_linear_step_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "z"],
            "speed": [LOW_SPEED, MED_SPEED]
        }
    },
    "angular_step": {
        "taskcase_generator": "generate_angular_step_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["r", "y"],
            "rpy0": [30.0, 45.0, 60.0],
            "speed": [LOW_SPEED, MED_SPEED]
        }
    },
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
            "ax": ["x", "y", "z"],
            "dh": [0.0, 1.0],
            "radii": [0.25, 3.0],
            "rdp_threshold": [0.1, 0.25],
            "res": [0.01, 0.05, 0.5],
            "speed": [LOW_SPEED, MED_SPEED],
            "trajectory_generator": [
                "cubic_spline", 
                "min_snap"
            ]
        }
    }
}


#################### AGGRESSIVE TASK BATTERY ####################
#################################################################


AGGRESSIVE_TASK_BATTERY = {
    "takeoff": {
        "taskcase_generator": "generate_takeoff_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "speed": [HIGH_SPEED]
        }
    },
    "landing": {
        "taskcase_generator": "generate_landing_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "speed": [HIGH_SPEED]
        }
    },
    "linear_step": {
        "taskcase_generator": "generate_linear_step_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "y", "z"],
            "speed": [HIGH_SPEED]
        }
    },
    "angular_step": {
        "taskcase_generator": "generate_angular_step_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["r", "p", "y"],
            "rpy0": [30.0, 45.0, 60.0],
            "speed": [HIGH_SPEED]
        }
    },
    "straight_away": {
        "taskcase_generator": "generate_straight_away_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "y", "z"],
            "speed": [HIGH_SPEED]
        }
    },
    "figure_eight": {
        "taskcase_generator": "generate_figure_eight_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x", "y", "z"],
            "dh": [0.0, 2.0],
            "radii": [0.25, 1.0, 4.0],
            "rdp_threshold": [0.025, 0.1, 0.25],
            "res": [0.05, 0.2],
            "speed": [MED_SPEED, HIGH_SPEED],
            "trajectory_generator": [
                "constant_speed", 
                "cubic_spline", 
                "min_snap"
            ]
        }
    }
}


#################### DEBUG TASK BATTERY ####################
############################################################


DEBUG_TASK_BATTERY = {
    "takeoff": {
        "taskcase_generator": "generate_takeoff_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "speed": [MED_SPEED]
        }
    },
    "landing": {
        "taskcase_generator": "generate_landing_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "speed": [MED_SPEED]
        }
    },
    "linear_step": {
        "taskcase_generator": "generate_linear_step_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x"],
            "speed": [MED_SPEED]
        }
    },
    "angular_step": {
        "taskcase_generator": "generate_angular_step_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["r", "y"],
            "rpy0": [30.0],
            "speed": [MED_SPEED]
        }
    },
    "straight_away": {
        "taskcase_generator": "generate_straight_away_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x"],
            "speed": [MED_SPEED]
        }
    },
    "figure_eight": {
        "taskcase_generator": "generate_figure_eight_tasks",
        "params": {
            "num_drones": NUM_DRONES,
            "ax": ["x"],
            "dh": [0.0, 2.0],
            "radii": [1.0, 4.0],
            "rdp_threshold": [0.025, 0.1, 0.25],
            "res": [0.05, 0.2],
            "speed": [HIGH_SPEED],
            "trajectory_generator": [
                "constant_speed", 
                "cubic_spline", 
                "min_snap"
            ]
        }
    }
}


#################### BATTERY ENUM ####################
######################################################


class TaskBattery(Enum):

    SMALL = SMALL_TASK_BATTERY

    FULL = FULL_TASK_BATTERY

    AGGRO = AGGRESSIVE_TASK_BATTERY

    DEBUG = DEBUG_TASK_BATTERY

    DEFAULT = DEBUG_TASK_BATTERY

DEFAULT_TASK_BATTERY = TaskBattery.FULL