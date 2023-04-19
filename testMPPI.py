import os
from scipy.spatial.transform import Rotation, RotationSpline
import matplotlib.pyplot as plt
from matplotlib import cm

import pybullet as pb
import numpy as np
import time
import pybullet_data

import mppi.cost_models as cost_models
import mppi.dynamics_models as dynamics_models
import mppi.MPPI_Node as MPPI_Node

##### Create MPPI Unit Test-Cases
config_fpath = os.path.join(os.getcwd(), "configs/mppi_config.json")
mppi_config = MPPI_Node.get_mppi_config(config_fpath)

INIT_XYZS = np.array([0., 0., 1.,])
INIT_RPYS = np.array([0., 0., 0.]) # np.pi/2.
INIT_STATE = np.hstack((INIT_XYZS, INIT_RPYS, np.zeros(6)))

TARGET_XYZS = np.array([0.,1., 1.,])
TARGET_RPYS = np.array([0., 0., 0.])
TARGET_STATE = np.hstack((INIT_XYZS, INIT_RPYS, np.zeros(6)))

mppi_node = MPPI_Node.MPPI(mppi_config)
# SAMPLE_TRAJECTORIES = np.zeros((mppi_config.T, mppi_config.K, mppi_config.X_SPACE))

T_horizon = np.linspace(0, mppi_config.T_HORIZON, num=mppi_config.T+1)
mppi_node.mppi_iter(INIT_STATE, TARGET_STATE)

fig, ax = plt.subplots(3, 1)

weight_threshold = 1e-8
filtered_sample_inds = np.nonzero(mppi_node.weights > weight_threshold)
filtered_weights = mppi_node.weights[filtered_sample_inds]
filtered_samples = mppi_node.SAMPLES_X[:, filtered_sample_inds, :]
best_sample = np.argmax(mppi_node.weights)
print(f"MPPI found {filtered_sample_inds.shape[0]} non-zero weighted samples out of {mppi_config.K}")

for k in range(filtered_weights.shape[0]):
    if k != best_sample:
        for i in range(3):
            ax[i].plot(T_horizon, mppi_node.SAMPLES_X[:, k, i], c=cm.jet(filtered_weights))
for i in range(3):
    ax[i].plot(T_horizon, mppi_node.SAMPLES_X[:, best_sample, i], c=cm.jet(mppi_node.weights[best_sample]), linewidth=2.0)

plt.show()

# cost_map_cmap = np.hstack([(mppi_node.COST_MAP - np.min(mppi_node.COST_MAP)) / (np.max(mppi_node.COST_MAP) - np.min(mppi_node.COST_MAP)) for t in range(mppi_config.T+1)])
# weight_cmap = np.vstack([mppi_node.weights  for t in range(mppi_config.T+1)]).flatten()
# plt.scatter(mppi_node.SAMPLES_X[:, :, 0], mppi_node.SAMPLES_X[:, :, 1], c=cm.jet(weight_cmap), edgecolor='none')

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.scatter(mppi_node.SAMPLES_X[:, :, 0], mppi_node.SAMPLES_X[:, :, 1], mppi_node.SAMPLES_X[:, :, 2], c=cm.jet(weight_cmap), edgecolor='none')
# plt.show()

# plt.plot(T_horizon, SAMPLE_TRAJECTORIES[:, :, 0])
# plt.plot(T_horizon, SAMPLE_TRAJECTORIES[:, :, 0])


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