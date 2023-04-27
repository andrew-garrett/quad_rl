import argparse
import os, sys
import time
from tqdm import tqdm
from collections import namedtuple
import wandb
try:
    import cupy as cp
except:
    print("Cupy not found, defaulting back to np/torch.")
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# from pytorch_mppi.mppi import MPPI # UM-ARM Lab's MPPI Implementation, used for reference

import mppi.MPPI_Node as MPPI_Node

class MPPITest_v2:
    """
    Class to simulate an iteration of MPPI, works for our np and torch versions, as well as pytorch_mppi
    """
    def __init__(self, mppi_config, mppi_node):
        ##### Get MPPI Config
        self.mppi_config = mppi_config
        self.T_HORIZON = np.linspace(0, self.mppi_config.T_HORIZON, num=self.mppi_config.T+1)
        self.dynamics_simulator = mppi_node.F
        ##### Create MPPI Object
        self.mppi_node = mppi_node
        self.reset()

    def reset(self):
        self.data = {
            "cycle_times": [],
            "tracked_trajectory": [],
            "optimal_rollout": [],
        }
        self.mppi_node.reset()

    def step(self, state, shift_nominal_trajectory=True):
        """
        This function should call an iteration to the MPPI Node.  If we are in the warmup phase (shift_nominal_trajectory = True),
        then it should just return nothing and do nothing.  On the first iteration where we pass the warmup phase, we take
        the current optimal control sequence and roll it out, computing the state-cost at each timestep of the full rollout.  This
        is our rollout-cost.  Then, we simulate taking the optimal control outputted by MPPI for one full time horizon and measure the
        state-cost on the final timestep.  This is our tracked-cost.
        """
        ##### Perform iteration of MPPI
        t_i = time.time()
        control_t = self.mppi_node.command(state, shift_nominal_trajectory)
        t_mppi = time.time() - t_i
        self.data["cycle_times"].append(t_mppi)

        ##### Get the current optimal control sequence
        smoothed_controls = self.mppi_node.U
        ##### In the warmup-phase, do nothing
        if not shift_nominal_trajectory:
            return
        
        # Apply control_t using the dynamics simulator and get it's cost
        state_tp1 = self.dynamics_simulator(state=state.reshape(1, -1), u=control_t.reshape(1, -1)) # size(1, 12)
        tracked_trajectory_cost = self.mppi_node.S.compute_state_cost(state_tp1)[0] # Scalar
        self.data["tracked_trajectory"].append((state_tp1, tracked_trajectory_cost)) # We track this, but we should focus on the TERMINAL COST

        ##### If we are out of the warmup phase, compute the cost of the current optimal control sequence (rollout-cost)
        if len(self.data["optimal_rollout"]) > 0:
            # Only collect the rollout-cost from the initial state
            return
        self.data["optimal_rollout"].append((state_tp1, tracked_trajectory_cost)) # Append the next-state and it's cost (because controls are rolling now)
        for i, ts_i in enumerate(self.T_HORIZON[1:]):
            state_tp1 = self.dynamics_simulator(state=state_tp1.reshape(1, -1), u=smoothed_controls[i].reshape(1, -1)) # size(1, 12)
            rollout_cost = self.mppi_node.S.compute_state_cost(state_tp1)[0] # Scalar
            self.data["optimal_rollout"].append((state_tp1, rollout_cost)) # Append the rollout next-state and it's cost
        return

    def simulate(self, initial_state, traj=None):
        """
        This function iterates through the full test.  First, warmup the MPPI trajectory for one full second of simulation time without shifting the 
        trajectory.  Then, on the first iteration after warming up, rollout the current optimal trajectory and compute its cost,
        focusing on the terminal cost, which is self.data["optimal_rollout"][-1][-1].  On this iteration and onwards, step MPPI and
        collect the tracked trajectory and it's cost, focusing on the terminal cost, which at the end of iteration will be
        self.data["tracked_trajectory"][-1][-1].
        """
        ##### Warm-up Phase
        for i in range(self.mppi_config.FREQUENCY):
            self.step(initial_state, shift_nominal_trajectory=False)
        
        ##### Rollout-cost collection Phase
        self.step(initial_state, shift_nominal_trajectory=True)

        ##### Tracked-cost collection Phase
        with tqdm(total=len(self.T_HORIZON), postfix=[""]) as tq:
            ##### For each trial in num_trials, run one full T_HORIZON of tracking
            for i, t in enumerate(self.T_HORIZON):
                state_t = self.data["tracked_trajectory"][-1][0]
                self.step(state_t, shift_nominal_trajectory=True)
                ##### TODO: Add functionality to accept a trajectory here
                if traj is not None:
                    target_state = traj.update(t+self.mppi_config.T_HORIZON)
                    self.mppi_node.reset(target_state)
                # Print Stats
                tq.set_postfix_str(self.print_stats())
                tq.update()
        ##### Process our results into proper numpy arrays
        tracked_trajectory = np.vstack(list(i[0] for i in self.data["tracked_trajectory"]))
        tracked_cost = np.array([i[1] for i in self.data["tracked_trajectory"]])
        rollout = np.vstack(list(i[0] for i in self.data["optimal_rollout"]))
        rollout_cost = np.array([i[1] for i in self.data["optimal_rollout"]])
        results_dict = {
            "tracked_trajectory": [(tracked_trajectory, tracked_cost)],
            "rollout": [(rollout, rollout_cost)],
            "mppi_hz": [1.0 / np.mean(self.data["cycle_times"])]
        }
        ##### Return our results
        return results_dict

    def run_trials(self, num_trials, init_state, target_state=None, traj=None):
        """
        Run through the same trajectory a few times.
        """
        self.mppi_node.reset(target_state)
        combined_results = {}
        for i in range(num_trials):
            results = self.simulate(init_state, traj=traj)
            for k, v in results.items():
                if k not in combined_results.keys():
                    combined_results[k] = v
                else:
                    combined_results[k].extend(v)
            self.reset()
        return combined_results

    def plot_results(self, results_dict, target_state=None, wandb_run=None):
        """
        Plot the optimal rollout and the tracked trajectory over time
        """
        tracked_trajectories = results_dict["tracked_trajectory"]
        rollouts = results_dict["rollout"]
        colors = ["r", "g", "b"]
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(projection="3d")

        # Initial Orientation
        drone_ax_i = Rotation.from_euler("xyz", tracked_trajectories[0][0][0, 3:6]).apply(0.1*np.eye(3))
        for j in range(3):
            d_ax = np.vstack([tracked_trajectories[0][0][0, :3], tracked_trajectories[0][0][0, :3] + drone_ax_i[j]]).T
            ax.plot(*d_ax, c=colors[j], linewidth=2)

        # Final Orientation
        drone_ax_f = Rotation.from_euler("xyz", tracked_trajectories[0][0][-1, 3:6]).apply(0.1*np.eye(3))
        for j in range(3):
            d_ax = np.vstack([tracked_trajectories[0][0][-1, :3], tracked_trajectories[0][0][-1, :3] + drone_ax_f[j]]).T
            ax.plot(*d_ax, c=colors[j], linewidth=2)
        
        # Tracked Trajectory
        ax.plot(*tracked_trajectories[0][0][:, :3].T, c="c", linewidth=2, label="Tracked Trajectory")
        for i in range(1, len(tracked_trajectories)):
            ax.plot(*tracked_trajectories[i][0][:, :3].T, c="c", linewidth=2)
            # Final Orientation
            drone_ax_f = Rotation.from_euler("xyz", tracked_trajectories[i][0][-1, 3:6]).apply(0.1*np.eye(3))
            for j in range(3):
                d_ax = np.vstack([tracked_trajectories[i][0][-1, :3], tracked_trajectories[i][0][-1, :3] + drone_ax_f[j]]).T
                ax.plot(*d_ax, c=colors[j], linewidth=2)
        
        # Rollout Trajectories
        ax.plot(*rollouts[0][0][:, :3].T, c="k", label="Optimal Trajectory")
        for i in range(1, len(rollouts)):
            ax.plot(*rollouts[i][0][:, :3].T, c="k")
        
        # Target State
        if target_state is not None:
            ax.scatter(*target_state[:3], c="k", label="Desired Position")
            # Target Orientation
            drone_ax_f = Rotation.from_euler("xyz", target_state[3:6]).apply(0.1*np.eye(3))
            for j in range(3):
                d_ax = np.vstack([target_state[:3], target_state[:3] + drone_ax_f[j]]).T
                ax.plot(*d_ax, c=colors[j], linewidth=2)

        ax.legend()
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_xlim3d(-1.5, 1.5)
        ax.set_ylim3d(-1.5, 1.5)
        ax.set_zlim3d(0, 1.5)
        if wandb_run is not None:
            fig.savefig("./trajectory.png")
            wandb.log({"Sample Trajectories": wandb.Image("./trajectory.png")})
        return fig, ax
        
    def log_results(self, results_dict, target_state=None, wandb_run=None):
        """
        Log the results, optionally to wandb
        """
        tracked_trajectories = results_dict["tracked_trajectory"]
        rollouts = results_dict["rollout"]
        mppi_hz = results_dict["mppi_hz"]

        if target_state is not None:
            mean_rollout_terminal_cost = np.mean([np.linalg.norm(r[0][-1] - target_state) for r in rollouts])
            mean_tracked_terminal_cost = np.mean([np.linalg.norm(t[0][-1] - target_state) for t in tracked_trajectories])
        else:
            mean_rollout_terminal_cost = np.mean([r[1][-1] for r in rollouts])
            mean_tracked_terminal_cost = np.mean([t[1][-1] for t in tracked_trajectories])
        mean_mppi_hz = np.mean(mppi_hz)

        log_dict = {
            "mean_rollout_terminal_cost": mean_rollout_terminal_cost,
            "mean_tracked_terminal_cost": mean_tracked_terminal_cost,
            "mean_mppi_hz": mean_mppi_hz
        }
        if wandb_run is not None:
            wandb.log(log_dict)
        else:
            print()
            for k, v in log_dict.items():
                print(f"{k}: {v}")
        
        return log_dict


    def print_stats(self):
        """
        Print Stats
        """
        freq_strs = str(round(1.0 / np.mean(self.data['cycle_times']), 3)).split(".")
        freq_str = ".".join([freq_strs[0].ljust(2, "0"), freq_strs[1].rjust(3, "0")])
        print_str = f"{freq_str} MPPI it/s"
        try: # For our MPPI method, record extra data
            good_samples_pct_strs = str(round(100.*self.data['good_samples_pct'][-1], 3)).split(".")
            good_samples_pct_str = ".".join([good_samples_pct_strs[0].ljust(2, "0"), good_samples_pct_strs[1].rjust(3, "0")])
            extra_str = f", {good_samples_pct_str}% of samples have weight >={round(self.weight_threshold, 3)}"
            print_str += extra_str
        except:
            pass
        return print_str
    


def evaluate(num_trials=3, sweep_config_path=None):
    """
    Evaluation of MPPI on a set of simple "trajectories"

    Logs to a WandB Sweep if sweep_config_path is provided
    """
    if sweep_config_path is None:
        # If not sweeping, just use current best parameters
        config_fpath = "./configs/mppi_config.json"
        wandb_run = None
    else:
        # Sets the parameters of the MPPI config according to wandb sweep
        config_fpath = sweep_config_path
        logging_config = {
            "reinit": True,
            "project": "ESE650 Final Project",
            "group": "Unit Trajectory Tuning"
        }
        wandb_run = wandb.init(**logging_config)

    ##### Initialize a fresh MPPI config, MPPI Object, and MPPITest Object
    mppi_config = MPPI_Node.get_mppi_config(config_fpath)
    mppi_node = MPPI_Node.MPPI(mppi_config, None)
    mppi_test_v2 = MPPITest_v2(mppi_config, mppi_node)

    ##### Initial State
    INIT_XYZS = np.array([0., 0., 1.,])
    INIT_RPYS = np.array([0., 0., 0.])
    INIT_STATE = np.hstack((INIT_XYZS, INIT_RPYS, np.zeros(6)))

    ##### Target States
    grid_positions = np.indices((3, 3, 3), dtype=np.float64)
    grid_positions = grid_positions.reshape(3, -1).T
    grid_positions -= 1
    TARGET_STATES = np.zeros((grid_positions.shape[0], 12))
    TARGET_STATES[:, :3] = grid_positions
    TARGET_STATES = TARGET_STATES[TARGET_STATES[:, 2] == INIT_XYZS[2], :]
    # TARGET_STATES[:, 6:8] = (TARGET_STATES[:, :2] - INIT_XYZS[:2]) / mppi_config.T_HORIZON # Set the target velocities
    TARGET_STATES[:, 6:8] = mppi_config.T_HORIZON # Set the target velocities
    
    logs = []
    verbose_inds = np.random.randint(0, TARGET_STATES.shape[0], 3)
    with tqdm(total=TARGET_STATES.shape[0]) as tq:
        for i, TARGET_STATE in enumerate(TARGET_STATES):
            if not np.all(TARGET_STATE[:3] == INIT_STATE[:3]):
                ##### For each target position, evaluate three different simple trajectories
                init_state_static = INIT_STATE.copy()
                init_state_45 = INIT_STATE.copy()
                init_state_45[5] = np.pi / 4.
                target_state_accel = TARGET_STATE.copy()
                target_state_accel[6:8] *= 2.
                init_target_pairs = [
                    (init_state_static, TARGET_STATE.copy()), # start at hover, end at velocity
                    (init_state_45, TARGET_STATE.copy()), # start at hover and yaw=45deg, end at velocity and yaw=0
                    (init_state_static, target_state_accel) # start at hover, end at 2*velocity
                ]
                for (init_state, target_state) in init_target_pairs:
                    combined_results = mppi_test_v2.run_trials(num_trials, init_state, target_state=target_state, traj=None)
                    if i in verbose_inds:
                        logs.append(mppi_test_v2.log_results(combined_results, target_state=target_state, wandb_run=wandb_run))
                        mppi_test_v2.plot_results(combined_results, target_state=target_state, wandb_run=wandb_run)
                    else:
                        logs.append(mppi_test_v2.log_results(combined_results, target_state=target_state))
            tq.update()
    
    ##### Log summary metrics
    summary_metrics = logs[0]
    for log_dict in logs[1:]:
        for k, v in log_dict.items():
            if type(summary_metrics[k]) is not list:
                summary_metrics[k] = [summary_metrics[k]]
            summary_metrics[k].append(v)
    for k, v in list(summary_metrics.items()):
        summary_metrics[k] = np.mean(v)
        # for k in logs[0].keys():
        summary_metrics[f"dataset_{k}"] = summary_metrics.pop(k)

    if sweep_config_path is not None:
        wandb.log(summary_metrics)
    else:
        for k, v in summary_metrics.items():
            print(f"{k}: {v}")

    return summary_metrics


                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPPI Testing Script, simple visualizer")
    parser.add_argument(
        "--sweep",
        help="config path",
        type=str
    )
    args = parser.parse_args()
        
    ##### Parameters
    num_trials = 5
    if args.sweep is not None:
        mppi_config = MPPI_Node.get_mppi_config(args.sweep)
    else:
        mppi_config = MPPI_Node.get_mppi_config()
    ##### Initial Pose
    INIT_XYZS = np.array([0., 0., 1.,])
    INIT_RPYS = np.array([0., 0., np.pi / 4.])
    ##### Target Pose
    TARGET_XYZS = np.array([1., 1., 1.,])
    TARGET_RPYS = np.array([0., 0., 0.])
    ##### Initial and Target Velocities
    # INIT_VS = np.array([TARGET_XYZS[0]/mppi_config.T_HORIZON, TARGET_XYZS[1]/mppi_config.T_HORIZON, 0.])
    INIT_VS = np.zeros(3)
    TARGET_VS = np.array([TARGET_XYZS[0]/mppi_config.T_HORIZON, TARGET_XYZS[1]/mppi_config.T_HORIZON, 0.])
    ##### Initial and Target States
    INIT_STATE = np.hstack((INIT_XYZS, INIT_RPYS, INIT_VS, np.zeros(3)))
    TARGET_STATE = np.hstack((TARGET_XYZS, TARGET_RPYS, TARGET_VS, np.zeros(3)))

    ##### Create an MPPI Object and MPPITest Object
    mppi_node = MPPI_Node.MPPI(mppi_config, TARGET_STATE)
    mppi_test_v2 = MPPITest_v2(mppi_config, mppi_node)
    
    ##### To plot a single simulation's results:
    # results = mppi_test_v2.simulate(INIT_STATE)
    # log = mppi_test_v2.log_results(results)
    # mppi_test_v2.plot_results(results, target_state=TARGET_STATE)
    
    ##### To plot trials with the same simulation parameters:
    combined_results = mppi_test_v2.run_trials(num_trials, INIT_STATE, target_state=TARGET_STATE, traj=None)
    fig, ax = mppi_test_v2.plot_results(combined_results, target_state=TARGET_STATE)
    plt.show()

    ##### Uncomment here to log the results to weights and biases
    # wandb_run = wandb.init()
    # mppi_test_v2.log_results(combined_results, wandb_run=wandb_run)
    # wandb.finish()






# v simple traj test
# from ..traj_gen import MinSnapTrajectoryGenerator
# dummy_config_dict = {
#     "speed": 2.0,
#     "rdp_threshold": 0.05
# }
# dummy_config = namedtuple("dummy_config", dummy_config_dict.keys())(dummy_config_dict)
# root, task_fname = "./bootstrap/atasets/dyn/DEBUG_001/", "task_figure_eight_ax-y_radii-1.0_dh-0.0_res-0.5.csv"
# traj = MinSnapTrajectoryGenerator(dummy_config, root=root, task_name=task_fname)
# combined_results = mppi_test_v2.run_trials(num_trials, INIT_STATE, traj=traj)
