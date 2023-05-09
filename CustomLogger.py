import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from cycler import cycler

from gym_pybullet_drones.utils.Logger import Logger


class CustomLogger(Logger):
    """A sub-class for more logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 logging_freq_hz: int,
                 output_folder: str="results",
                 num_drones: int=1,
                 duration_sec: int=0,
                 colab: bool=False,
                 ):
        """Logger class __init__ method.

        Note: the order in which information is stored by Logger.log() is not the same
        as the one in, e.g., the obs["id"]["state"], check the implementation below.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        super().__init__(logging_freq_hz,
                         output_folder,
                         num_drones,
                         duration_sec,
                         colab)
        ##### Log states, control targets, and flat trajectories
        self.optimal_trajectories = np.zeros((num_drones, 12, duration_sec*self.LOGGING_FREQ_HZ)) #### 12 control targets: pos_x,
                                                                                                                      # pos_y,
                                                                                                                      # pos_z,
                                                                                                                      # vel_x, 
                                                                                                                      # vel_y,
                                                                                                                      # vel_z,
                                                                                                                      # roll,
                                                                                                                      # pitch,
                                                                                                                      # yaw,
                                                                                                                      # ang_vel_x,
                                                                                                                      # ang_vel_y,
                                                                                                                      # ang_vel_z

    ################################################################################

    def log(self,
            drone: int,
            timestamp,
            state,
            control=np.zeros(12),
            optimal_trajectory=np.zeros(12)
            ):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        control : ndarray, optional
            (12,)-shaped array of floats containing the drone's control target.

        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != 20 or len(control) != 12 or len(optimal_trajectory) != 12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 16, 1))), axis=2)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
            self.optimal_trajectories = np.concatenate((self.optimal_trajectories, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = np.hstack([state[0:3], state[10:13], state[7:10], state[13:20]])
        self.controls[drone, :, current_counter] = control
        #### Keep track of the flat trajectories
        self.optimal_trajectories[drone, :, current_counter] = optimal_trajectory
        self.counters[drone] = current_counter + 1

    ################################################################################

    def evaluate(self, eval_fpath):
        """Evaluate the controller performance
        Metrics:
            - Tracking Error: How close does the drone track it's optimal/local trajectory (controller error)
            - Rollout Error: How close does the optimal/local trajectory come to the GLOBAL trajectory (planner error)
        """
        errors = {}
        state_mapping = ["p", "v", "q", "w"]
        for k in range(self.NUM_DRONES):
            for triple_ind in range(0, 10, 3):
                tracked = self.states[k, triple_ind:triple_ind+3]
                global_target = self.controls[k, triple_ind:triple_ind+3]
                optimal_target = self.optimal_trajectories[k, triple_ind:triple_ind+3]

                if triple_ind == 6: # Orientation Triple
                    ##### Tracking Error Statistics
                    raw_tracking_error = (Rotation.from_euler("xyz", tracked.T) * Rotation.from_euler("xyz", optimal_target.T).inv())
                    mean_raw_tracking_error = raw_tracking_error.mean().as_euler("xyz")
                    norm_tracking_error = raw_tracking_error.magnitude()
                    mean_norm_tracking_error = np.array(np.mean(norm_tracking_error, axis=0))
                    raw_tracking_error = raw_tracking_error.as_euler("xyz").T
                    ##### Rollout Error Statistics
                    raw_rollout_error = (Rotation.from_euler("xyz", global_target.T) * Rotation.from_euler("xyz", optimal_target.T).inv())
                    mean_raw_rollout_error = raw_rollout_error.mean().as_euler("xyz")
                    norm_rollout_error = raw_rollout_error.magnitude()
                    mean_norm_rollout_error = np.array(np.mean(norm_rollout_error, axis=0))
                    raw_rollout_error = raw_rollout_error.as_euler("xyz").T
                else:
                    ##### Tracking Error Statistics
                    raw_tracking_error = tracked - optimal_target
                    mean_raw_tracking_error = np.mean(raw_tracking_error, axis=1)
                    norm_tracking_error = np.linalg.norm(raw_tracking_error, axis=0)
                    mean_norm_tracking_error = np.array(np.mean(norm_tracking_error, axis=0))
                    ##### Rollout Error Statistics
                    raw_rollout_error = global_target - optimal_target
                    mean_raw_rollout_error = np.mean(raw_rollout_error, axis=1)
                    norm_rollout_error = np.linalg.norm(raw_rollout_error, axis=0)
                    mean_norm_rollout_error = np.array(np.mean(norm_rollout_error, axis=0))

                state_key = state_mapping[0]
                if state_key not in errors.keys():
                    errors[state_key] = {
                        "raw_tracking_error": [],
                        "mean_raw_tracking_error": [], 
                        "norm_tracking_error": [], 
                        "mean_norm_tracking_error": [], 
                        "raw_rollout_error": [],
                        "mean_raw_rollout_error": [],
                        "norm_rollout_error": [],
                        "mean_norm_rollout_error": []
                    }
                errors[state_key]["raw_tracking_error"].append(raw_tracking_error.reshape(1, 3, -1))
                errors[state_key]["mean_raw_tracking_error"].append(mean_raw_tracking_error.reshape(1, 3, -1)) 
                errors[state_key]["norm_tracking_error"].append(norm_tracking_error.reshape(1, 1, -1))
                errors[state_key]["mean_norm_tracking_error"].append(mean_norm_tracking_error.reshape(1, 1, -1)) 
                errors[state_key]["raw_rollout_error"].append(raw_rollout_error.reshape(1, 3, -1))
                errors[state_key]["mean_raw_rollout_error"].append(mean_raw_rollout_error.reshape(1, 3, -1))
                errors[state_key]["norm_rollout_error"].append(norm_rollout_error.reshape(1, 1, -1))
                errors[state_key]["mean_norm_rollout_error"].append(mean_norm_rollout_error.reshape(1, 1, -1))

                state_mapping = state_mapping[1:]
                state_mapping.append(state_key)
        

        for state_key in state_mapping:
            state_metrics = errors[state_key]
            for metric_key in state_metrics.keys():
                ##### Stack metrics along num_drones axis
                errors[state_key][metric_key] = np.vstack(errors[state_key][metric_key])
                ##### Prepare aggregated metrics keys
                if metric_key not in errors.keys():
                    errors[metric_key] = []
                errors[metric_key].append(errors[state_key][metric_key])
            errors.pop(state_key)
        for metric_key in errors.keys():
            ##### Aggregate metrics
            errors[metric_key] = np.hstack(errors[metric_key])

        ##### Save the eval data to npy file
        with open(eval_fpath, 'wb') as out_file:
            np.savez(out_file, **errors)

        return os.path.exists(eval_fpath)

    def save(self):
        """Save the logs and plots to file.
        """
        ##### Unwrap all yaw values before saving/plotting
        self.controls[:, 8, :] = np.unwrap(self.controls[:, 8, :], axis=1)
        self.states[:, 8, :] = np.unwrap(self.states[:, 8, :], axis=1)
        self.optimal_trajectories[:, 8, :] = np.unwrap(self.optimal_trajectories[:, 8, :], axis=1)
        trial_fpath = os.path.join(self.OUTPUT_FOLDER, "save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + ".npy")
        plot_fpath = trial_fpath.replace("sim_data", "plots").replace("npy", "png")
        eval_fpath = trial_fpath.replace(".npy", "_eval_data.npy")
        if not os.path.exists(os.path.dirname(trial_fpath)):
            os.makedirs(os.path.dirname(trial_fpath), exist_ok=True)
        ##### Save the flight data to npy file
        with open(trial_fpath, 'wb') as out_file:
            np.savez(out_file, timestamps=self.timestamps, states=self.states, controls=self.controls, trajectories=self.optimal_trajectories)
        ##### Save the flight data plot to png
        self.plot(plot_fpath=plot_fpath)
        self.evaluate(eval_fpath)

    ################################################################################
    
    def plot(self, pwm=False, plot_fpath=None):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        t = np.linspace(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, num=self.timestamps.shape[1])
        if t.shape[0] == 0:
            return
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(8, 2, figsize=(20, 10))

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 0, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 0, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 1, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 1, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 2, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 2, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 6, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 6, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 6, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 7, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 7, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, np.unwrap(self.states[j, 8, :]), label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 8, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 8, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (rad)')

        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 3, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 3, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 3, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 4, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 4, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 4, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 5, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 5, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 5, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### RPY Rates #############################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 9, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 9, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 10, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 10, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 11, :], label="drone_"+str(j)+"_des")
            axs[row, col].plot(t, self.optimal_trajectories[j, 11, :], label="drone_"+str(j)+"_opt")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz')

        ### This IF converts RPM into PWM for all drones ###########
        #### except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(12,16):
                if pwm and j > 0:
                    self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

        #### RPMs ##################################################
        col = 0
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 12, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('RPM0')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 13, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('RPM1')
        
        col = 1
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        #### Drawing options #######################################
        for i in range (8):
            for j in range (2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                         frameon=True
                         )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        if self.COLAB or plot_fpath is not None: 
            if not os.path.exists(os.path.dirname(plot_fpath)):
                os.makedirs(os.path.dirname(plot_fpath))
            plt.savefig(plot_fpath)
        else:
            plt.show()
        
    plt.close()
