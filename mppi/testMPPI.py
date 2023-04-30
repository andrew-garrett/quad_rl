import os
from copy import deepcopy
import time
from tqdm import tqdm
try:
    import cupy as cp
except:
    print("Cupy not found, defaulting back to np/torch.")
import numpy as np
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# from pytorch_mppi.mppi import MPPI # UM-ARM Lab's MPPI Implementation, used for reference

try:
    import mppi.cost_models as cost_models
    import mppi.dynamics_models as dynamics_models
    import mppi.MPPI_Node as MPPI_Node
except:
    import sys
    sys.path.append("../quad_rl")
    import cost_models as cost_models
    import dynamics_models as dynamics_models
    import MPPI_Node as MPPI_Node


class MPPITest:
    """
    Class to simulate an iteration of MPPI, works for our np and torch versions, as well as pytorch_mppi
    """
    def __init__(self, mppi_config, mppi_model):
        ##### Get MPPI Config
        self.mppi_config = mppi_config
        self.T_horizon = np.linspace(0, self.mppi_config.T_HORIZON, num=self.mppi_config.T+1)
        self.dynamics_simulator = mppi_model.F
        self.analytical_model = mppi_model.analytical_model
        ##### Create MPPI Object
        self.mppi_node = mppi_model
        self.stats = {"runtime": [], "good_samples_pct": []}

    def step(self, state, shift_nominal_trajectory=True, verbose=False):
        
        ##### Perform iteration of MPPI
        t_i = time.time()
        control_t = self.mppi_node.command(state, shift_nominal_trajectory)
        t_mppi = time.time() - t_i
        self.stats["runtime"].append(t_mppi)
        ##### Process outputs of MPPI

        # Simulate smoothed optimal controls and generate the approximated smooth optimal trajectory
        if shift_nominal_trajectory:
            state_tp1 = self.dynamics_simulator(state=deepcopy(state).reshape(1, -1), u=deepcopy(control_t).reshape(1, -1)).flatten()
            tracked_trajectory_cost = self.mppi_node.S.compute_state_cost(deepcopy(state_tp1))
        else:
            state_tp1 = deepcopy(state)
            tracked_trajectory_cost = 0.

        smoothed_controls = deepcopy(self.mppi_node.U)
        rollout_next_state = deepcopy(state_tp1)
        #analytical_output = deepcopy(state_tp1)
        optimal_trajectory = rollout_next_state
        #analytical_trajectory = analytical_output
        optimal_trajectory_cost = tracked_trajectory_cost

        for t, u_t in enumerate(smoothed_controls):
           #breakpoint()
            #analytical_accel = self.analytical_model.accelerationLabels(state=deepcopy(rollout_next_state).reshape(1, -1), u=deepcopy(u_t).reshape(1, -1))
            #nn_accel = self.dynamics_simulator.accelerationLabels(state=deepcopy(rollout_next_state).reshape(1, -1), u=deepcopy(u_t).reshape(1, -1))
            rollout_next_state = self.dynamics_simulator(state=deepcopy(rollout_next_state).reshape(1, -1), u=deepcopy(u_t).reshape(1, -1))
            #analytical_output = self.analytical_model(state=deepcopy(analytical_output).reshape(1, -1), u=deepcopy(u_t).reshape(1, -1))
            optimal_trajectory = self.mppi_config.METHOD.vstack((optimal_trajectory, deepcopy(rollout_next_state)))
            #analytical_trajectory = self.mppi_config.METHOD.vstack((analytical_trajectory, deepcopy(analytical_output)))
            optimal_trajectory_cost += self.mppi_node.S.compute_state_cost(deepcopy(rollout_next_state))
        
        # Put all of the results into numpy arrays before we exit the function so we don't have to post-process results
        if self.mppi_config.METHOD.__name__ == "torch":
            self.optimal_trajectory = optimal_trajectory.clone().cpu().numpy()
            state_tp1 = state_tp1.clone().cpu().numpy()
            control_t = control_t.clone().cpu().numpy()
        else:
            self.optimal_trajectory = optimal_trajectory.copy()
            state_tp1 = state_tp1.copy()
            control_t = control_t.copy()
            if self.mppi_config.METHOD.__name__ == "cupy":
                self.optimal_trajectory = cp.asnumpy(self.optimal_trajectory)
                state_tp1 = cp.asnumpy(state_tp1)
                control_t = cp.asnumpy(control_t)
        if verbose:
            try: # For our MPPI method, save other data
                if self.mppi_node.METHOD.__name__ == "torch": # this call will error for their MPPI method
                    weights = self.mppi_node.weights.clone().cpu().numpy()
                    samples = self.mppi_node.SAMPLES_X.clone().cpu().numpy()
                else:
                    weights = self.mppi_node.weights.copy()
                    samples = self.mppi_node.SAMPLES_X.copy()
                    if self.mppi_node.METHOD.__name__ == "cupy":
                        weights = cp.asnumpy(weights)
                        samples = cp.asnumpy(samples)

                # Filter out samples if their weights are below 1e-2, or just take the top 10 trajectories
                filtered_sample_inds = np.nonzero(~np.isnan(weights))[0]
                self.weight_threshold = 1e-2
                self.stats["good_samples_pct"].append(np.nonzero(weights[filtered_sample_inds] > self.weight_threshold)[0].shape[0] / self.mppi_config.K)
                filtered_sample_inds = np.argsort(weights[filtered_sample_inds])[-10:]
                filtered_weights = weights[filtered_sample_inds]
                self.filtered_samples = samples[:, filtered_sample_inds, :]
                self.best_sample_ind = np.argmax(filtered_weights) # find the best sample index
                # Transform the weights for color-mapping trajectories by weights
                self.filtered_weights = filtered_weights / np.sum(filtered_weights)
                # self.filtered_weights = (self.filtered_weights - np.min(self.filtered_weights)) / (np.max(self.filtered_weights) - np.min(self.filtered_weights))
            except:
                pass

        return state_tp1, control_t, tracked_trajectory_cost, optimal_trajectory_cost

    def print_stats(self):
        freq_strs = str(round(1.0 / np.mean(self.stats['runtime']), 3)).split(".")
        freq_str = ".".join([freq_strs[0].ljust(2, "0"), freq_strs[1].rjust(3, "0")])
        print_str = f"{freq_str} MPPI it/s"
        try: # For our MPPI method, record extra data
            good_samples_pct_strs = str(round(100.*self.stats['good_samples_pct'][-1], 3)).split(".")
            good_samples_pct_str = ".".join([good_samples_pct_strs[0].ljust(2, "0"), good_samples_pct_strs[1].rjust(3, "0")])
            extra_str = f", {good_samples_pct_str}% of samples have weight >={round(self.weight_threshold, 3)}"
            print_str += extra_str
        except:
            pass
        return print_str

def simulate(mppi_config, INIT_STATE, TARGET_STATE, verbose=False):
    """
    Simulates MPPI Control for T timesteps, and returns the total cost as sum(tracked_cost) + mean(optimal_cost)
    """
    if verbose:
        return simulate_and_plot(mppi_config, INIT_STATE, TARGET_STATE)
    # Create a new MPPI Object and MPPITest Object
    if mppi_config.METHOD.__name__ == "torch":
        state_tp1 = mppi_config.METHOD.asarray(deepcopy(INIT_STATE)).to(device=mppi_config.DEVICE, dtype=mppi_config.DTYPE)
        target_state = mppi_config.METHOD.asarray(deepcopy(TARGET_STATE)).to(device=mppi_config.DEVICE, dtype=mppi_config.DTYPE)
    else:
        state_tp1 = mppi_config.METHOD.asarray(deepcopy(INIT_STATE), dtype=mppi_config.DTYPE)
        target_state = mppi_config.METHOD.asarray(deepcopy(TARGET_STATE), dtype=mppi_config.DTYPE)
    mppi = MPPI_Node.MPPI(mppi_config, target_state)
    mppi_test = MPPITest(mppi_config, mppi)
    # For one T_horizon, simulate MPPI control
    optimal_trajectory_arr = []
    optimal_trajectory_cost = 0.0
    tracked_trajectory = [INIT_STATE]
    tracked_trajectory_cost = 0.0
    for i, ts_i in enumerate(range(mppi_config.T+1)):
        if i == 0:
            # Warm up the MPPI Trajectory before first action
            for j in range(10):
                mppi_test.step(state_tp1, shift_nominal_trajectory=False)
        state_tp1, control_t, tracked_trajectory_cost_t, optimal_trajectory_cost_t = mppi_test.step(state_tp1)
        optimal_trajectory_arr.append(deepcopy(mppi_test.optimal_trajectory))
        tracked_trajectory.append(deepcopy(state_tp1))
        tracked_trajectory_cost += tracked_trajectory_cost_t
        optimal_trajectory_cost += optimal_trajectory_cost_t

    # Return the total cost of the tracked trajectory (scalar)
    return tracked_trajectory_cost, optimal_trajectory_cost, None


def simulate_and_plot(mppi_config, INIT_STATE, TARGET_STATE):
    """
    Simulates MPPI Control for T timesteps, and returns the total cost as sum(tracked_cost) + mean(optimal_cost)
    Plots the results to a gif file
    """
    # Create a new MPPI Object and MPPITest Object
    if mppi_config.METHOD.__name__ == "torch":
        state_tp1 = mppi_config.METHOD.asarray(deepcopy(INIT_STATE)).to(device=mppi_config.DEVICE, dtype=mppi_config.DTYPE)
        target_state = mppi_config.METHOD.asarray(deepcopy(TARGET_STATE)).to(device=mppi_config.DEVICE, dtype=mppi_config.DTYPE)
    else:
        state_tp1 = mppi_config.METHOD.asarray(deepcopy(INIT_STATE), dtype=mppi_config.DTYPE)
        target_state = mppi_config.METHOD.asarray(deepcopy(TARGET_STATE), dtype=mppi_config.DTYPE)
    mppi = MPPI_Node.MPPI(mppi_config, target_state)
    mppi_test = MPPITest(mppi_config, mppi)
    # For one T_horizon, simulate MPPI control
    optimal_trajectory_arr = []
    optimal_trajectory_cost_arr = []
    tracked_trajectory = [INIT_STATE]
    tracked_trajectory_cost_arr = []
    filtered_weights_arr = []
    filtered_samples_arr = []
    T_multiplier = 1
    with tqdm(total=T_multiplier*(mppi_config.SIMULATION_T+1), postfix=[""]) as tq:
        for i, ts_i in enumerate(range(T_multiplier*(mppi_config.SIMULATION_T+1))):
            if i == 0:
                # Warm up the MPPI Trajectory before first action
                for j in range(5):
                    mppi_test.step(state_tp1, shift_nominal_trajectory=False, verbose=True)
            state_tp1, control_t, tracked_trajectory_cost_t, optimal_trajectory_cost_t = mppi_test.step(state_tp1, verbose=True)
            optimal_trajectory_arr.append(deepcopy(mppi_test.optimal_trajectory))
            tracked_trajectory.append(deepcopy(state_tp1))
            filtered_weights_arr.append(deepcopy(mppi_test.filtered_weights))
            filtered_samples_arr.append(deepcopy(mppi_test.filtered_samples))
            if i < mppi_config.SIMULATION_T+1:
                # Track the optimal trajectory and its cost
                # optimal_trajectory_arr.append(deepcopy(mppi_test.optimal_trajectory))
                optimal_trajectory_cost_arr.append(optimal_trajectory_cost_t)
                # Tracks the tracked trajectory and its cost
                # tracked_trajectory.append(deepcopy(state_tp1))
                tracked_trajectory_cost_arr.append(tracked_trajectory_cost_t)
                # Tracks the good samples for visualization
                # filtered_weights_arr.append(deepcopy(mppi_test.filtered_weights))
                filtered_samples_arr.append(deepcopy(mppi_test.filtered_samples))
            # Print Stats
            tq.set_postfix_str(mppi_test.print_stats())
            tq.update()

    optimal_trajectory_arr = np.array(optimal_trajectory_arr)
    optimal_trajectory_cost_arr = np.array(optimal_trajectory_cost_arr)
    tracked_trajectory = np.array(tracked_trajectory)
    tracked_trajectory_cost_arr = np.array(tracked_trajectory_cost_arr)
    filtered_weights_arr = np.array(filtered_weights_arr)
    filtered_samples_arr = np.array(filtered_samples_arr)
    plot_interval = int(1000. * np.mean(mppi_test.stats["runtime"]))
    plot_trajectories(
        INIT_STATE, 
        TARGET_STATE, 
        optimal_trajectory_arr, 
        tracked_trajectory, 
        # filtered_weights_arr=filtered_weights_arr, 
        # filtered_samples_arr=filtered_samples_arr, 
        plot_interval=plot_interval, 
        save=True
    )
    # Return the total cost of the tracked trajectory (scalar)
    return tracked_trajectory_cost_arr, optimal_trajectory_cost_arr, 1.0 / np.mean(mppi_test.stats["runtime"])


def plot_trajectories(INIT_STATE, TARGET_STATE, optimal_trajectory_arr, tracked_trajectory, filtered_weights_arr=None, filtered_samples_arr=None, plot_interval=30, save=False):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    tracked_trajectory_line, = ax.plot(*tracked_trajectory[0, :3].T, c="c", linewidth=2, label="Tracked Trajectory")
    ani_all = filtered_samples_arr is not None and filtered_weights_arr is not None
    if ani_all:
        filtered_weight_cmap = cm.jet(filtered_weights_arr)
        filtered_weight_cmap[:, :, -1] = filtered_weights_arr # set the alpha channel too
        n_best = filtered_weights_arr.shape[-1]
        filtered_samples_lines = ax.plot(*filtered_samples_arr[0, :, 0, :3].T, c=filtered_weight_cmap[0, 0])
        # filtered_samples_lines.extend([ax.plot(*filtered_samples_arr[0, :, j, :3].T, c=filtered_weight_cmap[0, j])[0] for j in range(1, n_best)])
        filtered_samples_lines.extend([ax.plot(*filtered_samples_arr[0, :, j, :3].T, c="g")[0] for j in range(1, n_best)])
    else:
        optimal_trajectory_line, = ax.plot(*optimal_trajectory_arr[0, :, :3].T, c="b", label="Optimal Trajectory")
    
    # ax.scatter(*INIT_STATE[:3], c="r", label="Initial Position")
    ax.scatter(*TARGET_STATE[:3], c="g", label="Desired Position")
    ax.legend()
    ax.set_title("MPPI Optimal Trajectory Generation and Tracking")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    # ax.set_xlim(-1.5, 1.5)
    # ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 1.5)
    ax.view_init(elev=30, azim=-135, roll=0)

    def animate(i):
        optimal_trajectory_line.set_data_3d(*optimal_trajectory_arr[i, :, :3].T)
        tracked_trajectory_line.set_data_3d(*tracked_trajectory[:i, :3].T)
        ax.view_init(elev=30, azim=-135 + (180*i / optimal_trajectory_arr.shape[0]), roll=0)
        return optimal_trajectory_line, tracked_trajectory_line

    def animate_all(i):
        tracked_trajectory_line.set_data_3d(*tracked_trajectory[:i, :3].T)
        ax.view_init(elev=30, azim=-135 + (180*i / optimal_trajectory_arr.shape[0]), roll=0)
        for j, line in enumerate(filtered_samples_lines):
            line.set_data_3d(filtered_samples_arr[i, :, j, :3].T)
            # line.set_color(filtered_weight_cmap[i, j])
        
        return tracked_trajectory_line, *filtered_samples_lines

    animation_fn = animate_all if ani_all else animate
    ani = FuncAnimation(fig, animation_fn, frames=list(range(1, optimal_trajectory_arr.shape[0])), blit=True, interval=plot_interval, repeat=not save)
    if save:
        ani.save("./trajectory.gif")
    else:
        plt.show()



if __name__ == "__main__":

    mppi_config = MPPI_Node.get_mppi_config()
    ##### Initial State
    INIT_XYZS = np.array([0., 0., 1.,])
    INIT_RPYS = np.array([0., 0., 0.])
    INIT_STATE = np.hstack((INIT_XYZS, INIT_RPYS, np.zeros(6)))

    ##### Initial State
    TARGET_XYZS = np.array([5., 5., 1.,])
    TARGET_RPYS = np.array([0., 0., 0.])
    TARGET_STATE = np.hstack((TARGET_XYZS, TARGET_RPYS, np.zeros(6)))

    tracked_trajectory_cost_arr, optimal_trajectory_cost_arr, _ = simulate_and_plot(mppi_config, INIT_STATE, TARGET_STATE)
    # simulate(mppi_config, INIT_STATE, TARGET_STATE)







































# import pybullet as pb
# import pybullet_data
# from scipy.spatial.transform import Rotation, RotationSpline

# # dynamics_attributes = ["mass", "lateral_friction", "local_inertia_diagonal",
# #                        "local_inertia_position", "local_inertial_position", 
# #                        "coeff_restitution", "coeff_rolling_friction",
# #                        "coeff_spinning_friction", "contact_damping",
# #                        "contact_stiffness", "body_type", "collision_margin"]
# # print("================================")
# # for i, dyn_att in enumerate(pb.getDynamicsInfo(r2d2Id, -1)):
# #     if dyn_att != -1:
# #         print(f"{dynamics_attributes[i]}: {dyn_att}")
# # print("================================")

# r2d2StartPos = [1,1,1] # elevated
# r2d2StartOrientation = pb.getQuaternionFromEuler([0,0,np.pi/4.])
# ax_map = ["x", "y", "z"]
# ang_map = ["roll", "pitch", "yaw"]
# damping_map = ["Damped", "Undamped"]

# t_dur = np.arange(0, 2.5, 1./240.)
# T = t_dur.shape[0]
# u_period_steps = 48
# dt = np.diff(t_dur, n=1, axis=0)
# p_world = np.zeros((2, T, 3))
# q_world = np.zeros((2, T, 4))
# v_world = np.zeros_like(p_world)
# w_world = np.zeros_like(p_world)

# for i, damping in enumerate(damping_map):
#     physicsClient = pb.connect(pb.GUI)  # or pb.DIRECT for non-graphical version
#     pb.setAdditionalSearchPath(pybullet_data.getDataPath())
#     pb.setGravity(0,0,0)
#     # load the robot
#     r2d2Id = pb.loadURDF("./gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf",r2d2StartPos, r2d2StartOrientation)
#     if damping == "Undamped":
#         pb.changeDynamics(r2d2Id, -1, linearDamping=0.0, angularDamping=0.0)
#     # set rotational velocity same direction as robot is facing.
#     # pb.resetBaseVelocity(r2d2Id, linearVelocity=(0, 0, 0), angularVelocity=(1, 1, 0))

#     for j, ts in enumerate(t_dur):
#         pb.stepSimulation()
#         time.sleep(1./240.)
#         # if j % u_period_steps == 0:
#         #     # pb.resetBaseVelocity(r2d2Id, linearVelocity=(0, 0, 0), angularVelocity=(1, 1, 0))
#         pb.applyExternalTorque(r2d2Id,
#                                 -1,
#                                 torqueObj=[0, 0, 0.00001],
#                                 flags = pb.URDF_USE_INERTIA_FROM_FILE
#                             )
#         r2d2Pos, r2d2Orn = pb.getBasePositionAndOrientation(r2d2Id)
#         r2d2Vel, r2d2AVel = pb.getBaseVelocity(r2d2Id)
        
#         p_world[i, j, :] = np.array(r2d2Pos)
#         q_world[i, j, :] = np.array(r2d2Orn)
#         v_world[i, j, :] = np.array(r2d2Vel)
#         w_world[i, j, :] = np.array(r2d2AVel)

#     pb.disconnect()

# ####################################################################################################

# # Create plots for position and orientation (euler angles)
# q_world_obj = Rotation.from_quat(q_world.reshape(-1, 4))
# q_world_euler = q_world_obj.as_euler("xyz").reshape(2, -1, 3)

# fig_pose, ax_pose = plt.subplots(3, 2)
# fig_pose.suptitle("Pose")
# ax_pose[0, 0].set_title("Position")
# ax_pose[0, 1].set_title("Orientation (Euler)")
# for i, damping in enumerate(damping_map):
#     for j in range(3):
#         ax_pose[j, 0].plot(p_world[i, :, j], label=damping)
#         ax_pose[j, 1].plot(q_world_euler[i, :, j], label=damping)
#         if i > 0:
#             ax_pose[j, 0].legend()
#             ax_pose[j, 1].legend()
#             continue
#         ax_pose[j, 0].set_ylabel(f"{ax_map[j]} (m)")
#         ax_pose[j, 1].set_ylabel(f"{ang_map[j]} (rad/s)")
    
# ####################################################################################################

# # Create plots for velocity (local frame and world frame)
# v_local = q_world_obj.apply(v_world.reshape(-1, 3)).reshape(2, -1, 3)

# fig_vel, ax_vel = plt.subplots(3, 2)
# fig_vel.suptitle("Velocity")
# ax_vel[0, 0].set_title("Global Frame")
# ax_vel[0, 1].set_title("Body Frame")
# for i, damping in enumerate(damping_map):
#     for j in range(3):
#         ax_vel[j, 0].plot(v_world[i, :, j], label=damping)
#         ax_vel[j, 1].plot(v_local[i, :, j], label=damping)
#         ax_vel[j, i].set_ylabel(f"v_{ax_map[j]} (m/s)")
#         if i > 0:
#             ax_pose[j, 0].legend()
#             ax_pose[j, 1].legend()

# ####################################################################################################

# # Create plots for velocity (local frame and world frame)
# lin_a_world = np.gradient(v_world.reshape(-1, 3), T, axis=0)
# lin_a_local = q_world_obj.apply(lin_a_world).reshape(2, -1, 3)
# lin_a_world = lin_a_world.reshape(2, -1, 3)

# fig_acc, ax_acc = plt.subplots(3, 2)
# fig_acc.suptitle("Accceleration")
# ax_acc[0, 0].set_title("Global Frames")
# ax_acc[0, 1].set_title("Body Frame")
# for i, damping in enumerate(damping_map):
#     for j in range(3):
#         ax_acc[j, 0].plot(lin_a_world[i, :, j], label=damping)
#         ax_acc[j, 1].plot(lin_a_local[i, :, j], label=damping)
#         ax_acc[j, i].set_ylabel(f"v_{ax_map[j]} (m/s)")
#         ax_acc[j, i].legend()
#         if i > 0:
#             ax_acc[j, 0].legend()
#             ax_acc[j, 1].legend()

# ####################################################################################################

# # Create plots for angular velocity (local frame and world frame) and delta(orientation)/delta(t)
# w_local = q_world_obj.apply(w_world.reshape(-1, 3)).reshape(2, -1, 3)

# # dq_world = Rotation.concatenate([q_world_obj[j+1] * q_world_obj[j].inv() for j in range(T - 1)])
# # dq_dt_world = dq_world.as_euler("xyz") / dt
# # dq_dt_world_np = np.gradient(q_world, [t_dur, t_dur, t_dur], axis=0) # Probably can't do this because its not necessarily valid for rotatations
# # q_world_spline_obj = RotationSpline(t_dur.flatten(), q_world_obj)
# # dq_dt_world_spline = q_world_spline_obj(t_dur.flatten(), order=1)

# fig_omega, ax_omega = plt.subplots(3, 2)
# fig_omega.suptitle("Angular Velocity")
# ax_omega[0, 0].set_title("Global Frame")
# ax_omega[0, 1].set_title("Body Frame")
# for i, damping in enumerate(damping_map):
#     for j in range(3):
#         ax_omega[j, 0].plot(v_world[i, :, j], label=damping)
#         ax_omega[j, 1].plot(v_local[i, :, j], label=damping)
#         ax_omega[j, i].set_ylabel(f"w_{ax_map[j]}, {ang_map[j]} (rad/s)")
#         if i > 0:
#             ax_omega[j, 0].legend()
#             ax_omega[j, 1].legend()

# ####################################################################################################

# # # Create plots for delta(angular_velocity)/delta(t)
# # dw_world = np.diff(w_world, n=1, axis=0)
# # dw_dt_world = dw_world / dt
# # dw_dt_world_np = np.gradient(w_world, [t_dur, t_dur, t_dur], axis=0)

# # # Create plots for delta(orientation)/delta(t)**2
# # dq_dt2_world = dq_dt_world / dt

# #################################################################################################### 

# # # The following quantities should be equal
# # r2d2AVel_local_mag = np.linalg.norm(r2d2AVel_local, axis=1)
# # r2d2AVel_world_mag = np.linalg.norm(r2d2AVel_world, axis=1)

# # fig, ax = plt.subplots(3, 2)
# # fig.suptitle("Angular Velocity")
# # ax[0, 0].set_title("Local Frame")
# # ax[0, 1].set_title("World Frame")

# # for i in range(3):
# #     ax[i, 0].plot(r2d2AVel_local[:, i])
# #     ax[i, 0].set_ylabel(ax_map[i])
# #     ax[i, 0].plot(r2d2AVel_local_mag)
# #     # ax[i, 0].legend()

# #     ax[i, 1].plot(r2d2AVel_world[:, i])
# #     ax[i, 1].set_ylabel(ax_map[i])
# #     ax[i, 1].plot(r2d2AVel_world_mag)
# #     # ax[i, 1].legend()

# # fig2, ax2 = plt.subplots(1, 2)
# # ax2[0].set_title("Local Frame")
# # ax2[0].plot(r2d2AVel_local_mag)
# # # ax2[0].legend()
# # ax2[1].set_title("World Frame")
# # ax2[1].plot(r2d2AVel_world_mag)
# # # ax2[1].legend()

# plt.show()