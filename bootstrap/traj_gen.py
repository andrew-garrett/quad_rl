#################### IMPORTS ####################
#################################################


import os
import json
import csv
from copy import deepcopy
from tqdm import tqdm
from itertools import product
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt

from task_gen import TRAJECTORY_PARAMS


#################### GLOBAL VARIABLES ####################
##########################################################


DEFAULT_ROOT = "./bootstrap/datasets/"

DEFAULT_DATASET_NAME = "dataset000"

DEFAULT_TASK_NAME = "linear_step.csv"

VERBOSE = True


#################### TRAJECTORY GENERATORS ####################
###############################################################


class TrajectoryGenerator:
    """
    Base Trajectory Generator Class
    """
    def __init__(self, params, root=DEFAULT_ROOT, task_name=DEFAULT_TASK_NAME):
        """
        Important variables/dimensions
        self.path           [N, 3]
        self.points         [self.n_wpts, 3]
        self.d_vecs         [self.n_wpts,]
        self.unit_vecs      [self.n_wpts, 3]
        self.t_start_vec    [self.n_wpts,1] or [self.n_wpts,]

        """
        self.root = root
        self.task_name = task_name
        self.params = params
        with open(f"{root}waypoints/{task_name}.csv", "r", newline="\n") as f:
            self.path = np.vstack(([row for i, row in enumerate(csv.reader(f)) if i > 0])).astype("float")

        # Generate Sparse Waypoints (only for complex paths)
        self.filter_waypoints()

        # Create time vector assuming constant speed trajectory
        self.vel_vec = self.params["speed"] * self.unit_vecs
        self.vel_vec[-1] *= 0.5
        try:
            self.t_start_vec = np.hstack(
                (np.zeros(1), np.cumsum(self.d_vecs / self.params["speed"], axis=0))
            )
        except:
            self.t_start_vec = np.vstack(
                (np.zeros(1), np.cumsum(self.d_vecs / self.params["speed"], axis=0))
            )

        self.is_done = False

        self.flat_output = { 'x': None, 'x_dot':None, 'x_ddot':None, 'x_dddot':None, 'x_ddddot':None,
                        'yaw':None, 'yaw_dot':None}


    def filter_waypoints(self):
        """
        Function to filter waypoints by:
            1. linear downsampling
            2. filtering by angle between consecutive downsampled waypoints
        """

        if "figure_eight" in self.task_name:
            # first linearly downsample according to a fixed rate
            subsample_inds = np.linspace(0, self.path.shape[0] - 1, int(self.params["res"]*self.path.shape[0]), dtype=int)
            subsampled_points = self.path[subsample_inds, :]
            # then perform rdp pruning
            self.points = self.rdp_filtering(subsampled_points)
        else:
            self.points = self.path
        self.n_wpts = self.points.shape[0]

        # d_vecs stores the (x,y,z) displacement between consecutive waypoints
        self.d_vecs = np.array(
            [
                np.linalg.norm(self.points[i + 1, :] - self.points[i, :]).flatten()
                if i < self.n_wpts - 1
                else np.zeros(1)
                for i in range(self.n_wpts)
            ]
        )
        # unit_vecs stores the unit vector between consecutive waypoints
        self.unit_vecs = np.array(
            [
                (self.points[i + 1, :] - self.points[i, :]) / self.d_vecs[i]
                if i < self.n_wpts - 1
                else np.zeros(3)
                for i in range(self.n_wpts)
            ]
        )


    def rdp_filtering(self, points):
        """
        Filters waypoints according to ramer-douglas-peucker algorithm
        """
        # Find maximum perpendicular distance from line formed along start and end of this segment
        max_dist, max_ind = 0, 0
        v = points[-1] - points[0]
        v_norm = np.linalg.norm(v)
        for i, pt in enumerate(points[1:-1], start=1):
            dist = np.linalg.norm(np.cross(pt - points[0], v)) / v_norm
            if (dist > max_dist):
                max_dist = dist
                max_ind = i
        
        if (max_dist > self.params["rdp_threshold"]):
            # Recursion
            return np.vstack((self.rdp_filtering(points[:max_ind])[:-1], self.rdp_filtering(points[max_ind:])))
        else:
            # Base Case
            return np.array([points[0], points[-1]])


    def get_trajectory_state(self, t):
        """
        Creates the state tuple for a constant speed trajectory
        """
        # Find the last waypoint that was traversed via t and t_start_vec
        t_diff = self.t_start_vec - t
        last_tstart_ind = np.argwhere(t_diff <= 0)[-1, 0]
        if last_tstart_ind == len(t_diff) - 1:
            x = self.points[-1]
            x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) for i in range(4))
            self.is_done = True
        else:
            t_progress = (t - self.t_start_vec[last_tstart_ind]) / (
                self.t_start_vec[last_tstart_ind + 1]
                - self.t_start_vec[last_tstart_ind]
            )
            x = (
                self.points[last_tstart_ind]
                + t_progress
                * self.d_vecs[last_tstart_ind]
                * self.unit_vecs[last_tstart_ind]
            )
            x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) if i > 0 else self.vel_vec[last_tstart_ind] for i in range(4))

        yaw, yaw_dot = 0.0, 0.0

        return x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot


    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        state = self.get_trajectory_state(t)

        for i, k in enumerate(self.flat_output.keys()):
            self.flat_output[k] = state[i]

        return self.flat_output.copy()
    

#################### B-SPLINE TRAJECTORY GENERATOR ####################
######################################################################


class BSplineTrajectoryGenerator(TrajectoryGenerator):
    """
    Sub-class to implement a k-th order interpolating B-Spline
    """
    def __init__(self, params, root=DEFAULT_ROOT, task_name=DEFAULT_TASK_NAME):

        if self.params["trajectory_generator"] == "cubic_spline":
            super().__init__(params, root, task_name)
            self.spline_order = 5
            self.spline_path = [
                make_interp_spline(
                    self.t_start_vec[:-1].flatten(), 
                    self.points[:, i], axis=0,
                    k=self.spline_order,
                    bc_type=(
                        [(j, 0.0) for j in range(1, 3)], 
                        [(j, 0.*self.params["speed"]) for j in range(1, 3)])) 
                for i in range(3)
            ]
        else:
            return TrajectoryGenerator(params, root, task_name)
        
    
    def get_trajectory_state(self, t):
        """
        Creates the state tuple for a basic k-th order spline interpolation trajectory
        """
        if np.any(np.isnan([self.spline_path[i](t, extrapolate=False) for i in range(3)])):
            self.is_done = True
            x = self.points[-1]
            x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) for i in range(4))
        elif np.any(np.isnan(np.array([self.spline_path[i](t+0.5, extrapolate=False) for i in range(3)]))):
            x = self.points[-1]
            x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) for i in range(4))
        else:
            x, x_dot, x_ddot, x_dddot, x_ddddot  = (
                np.array([
                    self.spline_path[i](t, nu=nu, extrapolate=False) 
                    for i in range(3)
                ]) for nu in range(self.spline_order)
            )
        
        yaw, yaw_dot = 0.0, 0.0
        
        return x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot
        

#################### MINIMUM SNAP TRAJECTORY GENERATOR ####################
###########################################################################


class MinSnapTrajectoryGenerator(TrajectoryGenerator):
    """
    Class to implement minimum snap trajectory generation
    """
    def __init__(self, params, root=DEFAULT_ROOT, task_name=DEFAULT_TASK_NAME):
        if self.params["trajectory_generator"] == "min_snap":
            super().__init__(params, root, task_name)
        else:
            return TrajectoryGenerator(params, root, task_name)

        # minimum snap
        num_coef = 8 * (self.n_wpts - 1)  # unknown c to solve for every segments
        A = np.zeros((num_coef, num_coef))
        b = np.zeros((num_coef, 3))
        index = 0  # keep records of rows

        # start
        A[0:4, 0:8] = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 2, 0, 0],
                                [0, 0, 0, 0, 6, 0, 0, 0]])
        b[0, :] = self.points[0]
        index += 4

        # update the matrix for each segment
        for i in range(self.n_wpts - 1):
            temp = self.t_start_vec[i]
            # end points
            if i == self.num_points - 2:
                A[num_coef - 4:num_coef, num_coef - 8:num_coef] = np.array([
                    [pow(temp, 7), pow(temp, 6), pow(temp, 5), pow(temp, 4), pow(temp, 3), pow(temp, 2), pow(temp, 1), 1],
                    [7 * pow(temp, 6), 6 * pow(temp, 5), 5 * pow(temp, 4), 4 * pow(temp, 3), 3 * pow(temp, 2), 2 * pow(temp, 1), 1, 0],
                    [42 * pow(temp, 5), 30 * pow(temp, 4), 20 * pow(temp, 3), 12 * pow(temp, 2), 6 * pow(temp, 1), 2, 0, 0],
                    [210 * pow(temp, 4), 120 * pow(temp, 3), 60 * pow(temp, 2), 24 * pow(temp, 1), 6, 0, 0, 0]])
                b[num_coef - 4] = self.points[i + 1]

            # points between start and end
            else:
                # total 8 rows
                # position constraints x1(t1) = x2(t1) = x1 2*
                A[index, 8 * i: 8 * (i + 1)] = np.array(
                    [[pow(temp, 7), pow(temp, 6), pow(temp, 5), pow(temp, 4), pow(temp, 3), pow(temp, 2), pow(temp, 1), 1]])
                A[index + 1, 8 * (i + 1): 8 * (i + 2)] = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])

                b[index] = self.points[i + 1]
                b[index + 1] = self.points[i + 1]

                # continuity constrain x1'(t1) = x2'(t1) 6*
                # velocity
                A[index + 2, 8 * i: 8 * (i + 2)] = np.array(
                    [[7 * pow(temp, 6), 6 * pow(temp, 5), 5 * pow(temp, 4), 4 * pow(temp, 3), 3 * pow(temp, 2), 2 * pow(temp, 1), 1, 0,
                      0, 0, 0, 0, 0, 0, -1, 0]])

                # acceleration
                A[index + 3, 8 * i: 8 * (i + 2)] = np.array([
                    [42 * pow(temp, 5), 30 * pow(temp, 4), 20 * pow(temp, 3), 12 * pow(temp, 2), 6 * pow(temp, 1), 2, 0, 0,
                     0, 0, 0, 0, 0, -2, 0, 0]])

                # jerk
                A[index + 4, 8 * i: 8 * (i + 2)] = np.array([
                    [210 * pow(temp, 4), 120 * pow(temp, 3), 60 * pow(temp, 2), 24 * pow(temp, 1), 6, 0, 0, 0,
                     0, 0, 0, 0, -6, 0, 0, 0]])

                # snap
                A[index + 5, 8 * i: 8 * (i + 2)] = np.array([
                    [840 * pow(temp, 3), 360 * pow(temp, 2), 120 * pow(temp, 1), 24, 0, 0, 0, 0,
                     0, 0, 0, -24, 0, 0, 0, 0]])

                # crackle
                A[index + 6, 8 * i: 8 * (i + 2)] = np.array([
                    [2520 * pow(temp, 2), 720 * pow(temp, 1), 120, 0, 0, 0, 0, 0,
                     0, 0, -120, 0, 0, 0, 0, 0]])

                # pop
                A[index + 7, 8 * i: 8 * (i + 2)] = np.array([
                    [5040 * pow(temp, 1), 720, 0, 0, 0, 0, 0, 0,
                     0, -720, 0, 0, 0, 0, 0, 0]])

                index += 8

        Cx = lsmr(A, b[:, 0])[0]
        Cy = lsmr(A, b[:, 1])[0]
        Cz = lsmr(A, b[:, 2])[0]
        self.c = np.hstack((Cx.reshape(len(A), 1), Cy.reshape(len(A), 1), Cz.reshape(len(A), 1))).reshape(-1, 8, 3)

        
    def get_trajectory_state(self, t):
        """
        Get the minimum snap state
        """
        # for i in range(self.num_points - 1):
        #     if t == 0:  # start
        #         x = self.points[0]
        #     elif t >= time[-1]:  # end
        #         x = self.points[-1]
        #     elif time[i] < t <= time[i + 1]:
        #         t_s = float(t - time[i])
        #         c = self.c[i]

        #         x = (np.array([pow(t_s, 7), pow(t_s, 6), pow(t_s, 5), pow(t_s, 4), pow(t_s, 3), pow(t_s, 2), pow(t_s, 1), 1]) @ c)
        #         x_dot = (np.array([7 * pow(t_s, 6), 6 * pow(t_s, 5), 5 * pow(t_s, 4), 4 * pow(t_s, 3), 3 * pow(t_s, 2), 2 * pow(t_s, 1), 1, 0]) @ c)
        #         x_ddot = (np.array([42 * pow(t_s, 5), 30 * pow(t_s, 4), 20 * pow(t_s, 3), 12 * pow(t_s, 2), 6 * pow(t_s, 1), 2, 0, 0]) @ c)
        #         x_dddot = (np.array([210 * pow(t_s, 4), 120 * pow(t_s, 3), 60 * pow(t_s, 2), 24 * pow(t_s, 1), 6, 0, 0, 0]) @ c)
        #         x_ddddot = (np.array([840 * pow(t_s, 3), 360 * pow(t_s, 2), 120 * pow(t_s, 1), 24, 0, 0, 0, 0]) @ c)
        return (np.zeros((3,1)) if i < 5 else 0. for i in range(7))


######################## UTILITY FUNCTIONS ########################
###################################################################


def yield_all_task_trajectories(root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME, verbose=False):
    """
    Generator function to yield TrajectoryGenerator Objects
    """
    for task_name in tqdm(sorted(os.listdir(f"{root}{dataset_name}/waypoints/"))):
        # get the parameters for the current task
        params = get_task_params(root, dataset_name, task_name[:-4])
        # for parameters that we must control at trajectory generation time,
        # we create a grid in the same way that we do in task_gen.py
        traj_search_params = []
        for k, v in sorted(params.items(), key=lambda x: x[0]):
            if type(v) == list:
                traj_search_params.append(k)
        traj_param_grid = product(*[params[k] for k in traj_search_params])
        # iterate through the trajectory specific parameter grid
        for i, traj_task_params in enumerate(traj_param_grid):
            # set the traj_search_paramss to their values in the current grid-search
            for j, key in enumerate(traj_search_params):
                params[key] = traj_task_params[j]
            if verbose:
                # display progress
                print(f"Creating Trajectory Object for {task_name[:-4]}", end="\r", flush=True)

            # yield the task_name, current params, and the corresponding Trajectory Generator Object
            yield (
                params,
                task_name[:-4],  
                TrajectoryGenerator(deepcopy(params), f"{root}{dataset_name}/", task_name[:-4])
            )


def collect_task_trajectory(task_trajectories, traj, t_vector, T):
    """
    Function for running one trial of simulating a trajectory
    """
    warmdown_iters = 30
    # tracking waypoints
    if "waypoints" not in task_trajectories.keys():
        task_trajectories["waypoints"] = {
            "sparse": [traj.points],
            "dense": [traj.path]
        }
    else: 
        task_trajectories["waypoints"]["sparse"].append(traj.points)
        task_trajectories["waypoints"]["dense"].append(traj.path)

    # simulate the trajectory for T timesteps
    x_traj = np.zeros((1, T, 3))                                                                                                                             
    x_dot_traj = np.zeros((1, T, 3))
    for j, t in enumerate(t_vector):
        # for each timestep, record the state
        for k, v in traj.update(t).items():
            if k == "x":
                x_traj[:, j, :] = v
            elif k == "x_dot":
                x_dot_traj[:, j, :] = v
        if traj.is_done:
            # run 30 extra timesteps to allow system to stabilize/warmdown
            warmdown_iters -= 1
            if warmdown_iters == 0:
                break
    if "x" not in task_trajectories.keys():
        task_trajectories["x"] = x_traj
        task_trajectories["x_dot"] = x_dot_traj
    else:
        task_trajectories["x"] = np.vstack((task_trajectories["x"], x_traj))
        task_trajectories["x_dot"] = np.vstack((task_trajectories["x_dot"], x_dot_traj))
    
    return task_trajectories


def get_task_params(root=DEFAULT_ROOT, dataset_name=DEFAULT_DATASET_NAME, task_name=DEFAULT_TASK_NAME):
    """
    Prepare parameters for planning a trajectory
    """

    # load params
    with open(f"{root}{dataset_name}/{dataset_name}.json", "r") as f:
        params_raw = json.load(f)

    params_prepped = {}
    for k in params_raw.keys():
        # find the taskcase params for task_name
        if k in task_name:
            params_prepped = params_raw[k]
            param_pairs = task_name[len(f"task_{k}_"):].split("_")
            for param_pair in param_pairs:
                # set list-based parameters to a scalar
                key, value = param_pair.split("-")
                try:
                    # handles numerical values
                    params_prepped[key] = float(value)
                except:
                    # handles string values
                    params_prepped[key] = value
            break
    
    return params_prepped


def plot_trajectories_by_task(task_group_trajectory_dict, task_group=DEFAULT_TASK_NAME):
    """
    Function to plot waypoints and trajectories from a given group of tasks
    """
    waypoints = task_group_trajectory_dict[task_group]["waypoints"]
    x_traj = task_group_trajectory_dict[task_group]["x"]
    x_dot_traj = task_group_trajectory_dict[task_group]["x_dot"]
    speed_traj = np.linalg.norm(x_dot_traj, axis=-1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # WAYPOINTS
    dense_wpts, sparse_wpts = waypoints["dense"], waypoints["sparse"]
    n_paths = len(dense_wpts)
    for i in range(0, n_paths, 10):
        dense_path = dense_wpts[i].T
        sparse_path = sparse_wpts[i].T
        dense_path[0, :] += i // 5
        sparse_path[0, :] += i // 5
        ax.plot(*dense_path, color="k", label="dense path", linewidth=1)
        ax.scatter(*sparse_path, s=100, label="sparse waypoints", marker="o")
    # TRAJECTORIES
    for i in range(0, n_paths, 10):
        traj = x_traj[i].T
        traj[0, :] += i // 5
        p = ax.scatter(*traj, c=speed_traj[i]/np.max(speed_traj[i]), cmap="jet", label="trajectory")
    # fig.colorbar(p)
    ax.set_title(f"{task_group}")
    # ax.legend()

    plt.show()
    plt.close()


#################### RUNNER ####################
################################################

def main():
    trajectories_by_task = {}
    t_vector = np.arange(0, 2.*np.pi, 0.01)
    T = t_vector.shape[0]
    prev_task_group = None
    # iterate through all trajectories
    for i, (params, task, traj_gen_obj) in enumerate(yield_all_task_trajectories(verbose=VERBOSE)):
        task_group = "_".join(task.split("_")[1:3])
        # if we have reached a new group of tasks, then
        if task_group not in trajectories_by_task.keys():
            if prev_task_group is not None:
                # for each trajectory in the previous group of tasks,
                print("", flush=True)
                for traj in trajectories_by_task[prev_task_group]["generated_trajectories"]:
                    trajectories_by_task[prev_task_group] = collect_task_trajectory(
                                                                            trajectories_by_task[prev_task_group], 
                                                                            traj, 
                                                                            t_vector, 
                                                                            T
                                                                        )
                # Plot Data and free some memory
                if VERBOSE:
                    plot_trajectories_by_task(trajectories_by_task, prev_task_group)
                trajectories_by_task[prev_task_group]["generated_trajectories"] = None
            
            # then, initialize the next group of tasks
            trajectories_by_task[task_group] = {
                "generated_trajectories": []
            }
        prev_task_group = task_group
        trajectories_by_task[task_group]["generated_trajectories"].append(traj_gen_obj)


if __name__ == "__main__":
    main()
    


