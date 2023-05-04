import os
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.Logger import Logger

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        self.flat_trajectories = np.zeros((num_drones, 17, duration_sec*self.LOGGING_FREQ_HZ)) #### 12 control targets: pos_x,
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
            flat_trajectory=np.zeros(17)
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
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != 20 or len(control) != 12 or len(flat_trajectory) != 17:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 16, 1))), axis=2)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
            self.flat_trajectories = np.concatenate((self.flat_trajectories, np.zeros((self.NUM_DRONES, 17, 1))), axis=2)
        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = np.hstack([state[0:3], state[10:13], state[7:10], state[13:20]])
        self.controls[drone, :, current_counter] = control
        #### Keep track of the flat trajectories
        self.flat_trajectories[drone, :, current_counter] = flat_trajectory
        self.counters[drone] = current_counter + 1

    ################################################################################

    def save(self):
        """Save the logs and plots to file.
        """
        trial_fpath = os.path.join(self.OUTPUT_FOLDER, "save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + ".npy")
        plot_fpath = trial_fpath.replace("sim_data", "plots").replace("npy", "png")
        if not os.path.exists(os.path.dirname(trial_fpath)):
            os.makedirs(os.path.dirname(trial_fpath), exist_ok=True)
        if not os.path.exists(os.path.dirname(plot_fpath)):
            os.makedirs(os.path.dirname(plot_fpath))
        ##### Save the flight data to npy file
        with open(trial_fpath, 'wb') as out_file:
            np.savez(out_file, timestamps=self.timestamps, states=self.states, controls=self.controls, trajectories=self.flat_trajectories)
        ##### Save the flight data plot to png
        self.plot(plot_fpath=plot_fpath)

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
        fig, axs = plt.subplots(10, 2, figsize=(20, 10))

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 0, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 1, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 2, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 6, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 6, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 7, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, np.unwrap(self.states[j, 8, :]), label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 8, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (rad)')

        #### Ang Vel ###############################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 9, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 10, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 11, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz')

        #### Time ##################################################
        row = 9
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')

        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 3, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 3, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 4, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 4, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 5, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 5, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### RPY Rates #############################################
        row = 3
        for j in range(self.NUM_DRONES):
            rdot = np.hstack([0, (self.states[j, 6, 1:] - self.states[j, 6, :-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, rdot, label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 9, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('rdot (rad/s)')
        row = 4
        for j in range(self.NUM_DRONES):
            pdot = np.hstack([0, (self.states[j, 7, 1:] - self.states[j, 7, :-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, pdot, label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 10, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('pdot (rad/s)')
        row = 5
        for j in range(self.NUM_DRONES):
            ydot = np.hstack([0, (np.unwrap(self.states[j, 8, 1:]) - np.unwrap(self.states[j, 8, :-1])) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, ydot, label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 11, :], label="drone_"+str(j)+"des")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('ydot (rad/s)')

        ### This IF converts RPM into PWM for all drones ###########
        #### except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(12,16):
                if pwm and j > 0:
                    self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

        #### RPMs ##################################################
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
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        #### Drawing options #######################################
        for i in range (10):
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
            plt.savefig(plot_fpath)
        else:
            plt.show()
        
    plt.close()