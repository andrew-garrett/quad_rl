#################### IMPORTS ####################
#################################################


import os
import csv
from copy import deepcopy
from tqdm import tqdm
from itertools import product
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.sparse.linalg import lsmr
from scipy.optimize import minimize

from bootstrap.task_gen import TRAJECTORY_PARAMS, DEFAULT_T, DEFAULT_DATASET_NAME, DEFAULT_ROOT, DEFAULT_TASK_BATTERY
from bootstrap.utils import collect_task_trajectory, get_task_params, plot_trajectories_by_task

#################### GLOBAL VARIABLES ####################
##########################################################


DEFAULT_TASK_NAME = "linear_step.csv"

VERBOSE = True


#################### TRAJECTORY GENERATORS ####################
###############################################################


class TrajectoryGenerator:
    """
    Base Trajectory Generator Class (Implements Constant Speed Trajectory)
    """
    def __init__(self, 
                 params, 
                 root=DEFAULT_ROOT, 
                 task_name=DEFAULT_TASK_NAME
    ):
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
        self.vel_vec[0] *= 0.5
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
        Filter waypoints by:
            1. linear downsampling
            2. filtering by angle between consecutive downsampled waypoints

        """

        if "figure_eight" in self.task_name:
            try:
                # try rdp filtering
                self.points = self.rdp_filtering(self.path)
            except:
                # if that fails, linear downsample
                self.points = self.path[::5, :]
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

        Parameters:
            - points: np.array(N, 3) - All (x, y, z) points in a dense path
        
        Returns:
            - filtered_waypoints: np.array(self.n_wpts, 3) - All (x, y, z) points in filtered path

        """
        # Find maximum perpendicular distance from line formed along start and end of this segment
        max_dist, max_ind = 0, 0
        v = points[-1] - points[0]
        v_norm = np.linalg.norm(v)
        orth_dists = np.linalg.norm(np.cross(points[1:-1] - points[0], v), axis=1) / v_norm
        max_dist, max_ind = np.max(orth_dists), np.argmax(orth_dists)
        
        if (max_dist > self.params["rdp_threshold"]):
            # Recursion
            return np.vstack((self.rdp_filtering(points[:max_ind])[:-1], self.rdp_filtering(points[max_ind:])))
        else:
            # Base Case
            return np.array([points[0], points[-1]])


    def get_trajectory_state(self, t):
        """
        Creates the state tuple for a constant speed trajectory

        Parameters:
            - t: float - the timestamp in seconds

        Returns:
            - (x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot): tuple(np.array(3,) for i in range(5), float, float) - The desired
            state for the current timestamp

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

        Parameters:
            - t: float - timestamp in seconds
        
        Returns:
            - flat_output: dict() - a dict describing the present desired flat outputs with keys
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
            if k == "x_dot":
                # scale the velocity?
                self.flat_output[k] = np.clip(state[i], -self.params["speed"], self.params["speed"])
            else:
                self.flat_output[k] = state[i]

        return self.flat_output.copy()
    

#################### HOVER TRAJECTORY GENERATOR ####################
####################################################################


# class HoverTrajectoryGenerator(TrajectoryGenerator):
#     """
#     Sub-class of TrajectoryGenerator to implement a stabilizing
#     """
#     def __init__(self, 
#                  params, 
#                  root=DEFAULT_ROOT, 
#                  task_name=DEFAULT_TASK_NAME
#     ):
        


#################### B-SPLINE TRAJECTORY GENERATOR ####################
######################################################################


class BSplineTrajectoryGenerator(TrajectoryGenerator):
    """
    Sub-class of TrajectoryGenerator to implement a k-th order interpolating B-Spline
    """
    def __init__(self, 
                 params, 
                 root=DEFAULT_ROOT, 
                 task_name=DEFAULT_TASK_NAME
    ):
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
        
    
    def get_trajectory_state(self, t):
        """
        Creates the state tuple for a basic k-th order spline interpolation trajectory

        Returns:
            - (x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot): tuple(np.array(3,) for i in range(5), float, float) - The desired
            state for the current timestamp

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
    Sub-class of TrajectoryGenerator to implement minimum snap trajectory generation
    """
    def __init__(self, 
                 params, 
                 root=DEFAULT_ROOT, 
                 task_name=DEFAULT_TASK_NAME
    ):
        super().__init__(params, root, task_name)

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
            temp = self.t_start_vec.flatten()[i] #/ self.t_start_vec.flatten()[-1]
            # end points
            if i == self.n_wpts - 2:
                A[num_coef - 4:num_coef, num_coef - 8:num_coef] = np.array([
                    [temp**7, temp**6, temp**5, temp**4, temp**3, temp**2, temp, 1],
                    [7 * temp**6, 6 * temp**5, 5 * temp**4, 4 * temp**3, 3 * temp**2, 2 * temp, 1, 0],
                    [42 * temp**5, 30 * temp**4, 20 * temp**3, 12 * temp**2, 6 * temp, 2, 0, 0],
                    [210 * temp**4, 120 * temp**3, 60 * temp**2, 24 * temp, 6, 0, 0, 0]])
                b[num_coef - 4] = self.points[i + 1]

            # points between start and end
            else:
                # total 8 rows
                # position constraints x1(t1) = x2(t1) = x1 2*
                A[index, 8 * i: 8 * (i + 1)] = np.array(
                    [[temp**7, temp**6, temp**5, temp**4, temp**3, temp**2, temp, 1]])
                A[index + 1, 8 * (i + 1): 8 * (i + 2)] = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])

                b[index] = self.points[i + 1]
                b[index + 1] = self.points[i + 1]

                # continuity constrain x1'(t1) = x2'(t1) 6*
                # velocity
                A[index + 2, 8 * i: 8 * (i + 2)] = np.array(
                    [[7 * temp**6, 6 * temp**5, 5 * temp**4, 4 * temp**3, 3 * temp**2, 2 * temp, 1, 0,
                      0, 0, 0, 0, 0, 0, -1, 0]])

                # acceleration
                A[index + 3, 8 * i: 8 * (i + 2)] = np.array([
                    [42 * temp**5, 30 * temp**4, 20 * temp**3, 12 * temp**2, 6 * temp, 2, 0, 0,
                     0, 0, 0, 0, 0, -2, 0, 0]])

                # jerk
                A[index + 4, 8 * i: 8 * (i + 2)] = np.array([
                    [210 * temp**4, 120 * temp**3, 60 * temp**2, 24 * temp, 6, 0, 0, 0,
                     0, 0, 0, 0, -6, 0, 0, 0]])

                # snap
                A[index + 5, 8 * i: 8 * (i + 2)] = np.array([
                    [840 * temp**3, 360 * temp**2, 120 * temp, 24, 0, 0, 0, 0,
                     0, 0, 0, -24, 0, 0, 0, 0]])

                # crackle
                A[index + 6, 8 * i: 8 * (i + 2)] = np.array([
                    [2520 * temp**2, 720 * temp, 120, 0, 0, 0, 0, 0,
                     0, 0, -120, 0, 0, 0, 0, 0]])

                # pop
                A[index + 7, 8 * i: 8 * (i + 2)] = np.array([
                    [5040 * temp, 720, 0, 0, 0, 0, 0, 0,
                     0, -720, 0, 0, 0, 0, 0, 0]])

                index += 8

        # Cx = minimize(lambda x: np.sum(A@x - b[:, 0]), np.zeros(num_coef), options={"maxiter": 100}) #, method="SLSQP")
        # Cy = minimize(lambda x: np.sum(A@x - b[:, 1]), np.zeros(num_coef), options={"maxiter": 100})
        # Cz = minimize(lambda x: np.sum(A@x - b[:, 2]), np.zeros(num_coef), options={"maxiter": 100})
        damping, atol, btol, maxiter = 0.1, 1e-5, 1e-4, 100000
        Cx = lsmr(A, b[:, 0], damping=damping, atol=atol, btol=btol, maxiter=maxiter)
        Cy = lsmr(A, b[:, 1], damping=damping, atol=atol, btol=btol, maxiter=maxiter)
        Cz = lsmr(A, b[:, 2], damping=damping, atol=btol, btol=btol, maxiter=maxiter)
        # if np.any([Cx[1] == 1, Cy[1] == 1, Cz[1] == 1]):
        print("+++++++++++++++++++++++++++ MIN SNAP TRAJECTORY ")
        print(self.task_name)
        for c in [Cx, Cy, Cz]:
            x, istop, itn, normr = c
            print(f"stop_code: {istop}, after {itn} iteration, with error {normr}")
        print()
        print()
        self.c = np.hstack((Cx[0].reshape(len(A), 1), Cy[0].reshape(len(A), 1), Cz[0].reshape(len(A), 1))).reshape(-1, 8, 3)

        
    def get_trajectory_state(self, t):
        """
        Creates the state tuple for a minimum snap trajectory

        Returns:
            - (x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot): tuple(np.array(3,) for i in range(5), float, float) - The desired
            state for the current timestamp

        """
        t_diff = self.t_start_vec - t
        last_tstart_ind = np.argwhere(t_diff <= 0)[-1, 0]
        if last_tstart_ind >= len(t_diff) - 1:
            self.is_done = True
            last_tstart_ind = len(t_diff) - 3
        # else:
        t_s = float(t - self.t_start_vec[last_tstart_ind]) #/ self.t_start_vec.flatten()[-1]
        c = self.c[last_tstart_ind]

        x = (np.array([t_s**i for i in range(7, -1, -1)]) @ c)
        x_dot = (np.array([(i)*t_s**max(0, i-1) for i in range(7, -1, -1)]) @ c)
        x_ddot = (np.array([(i*(i-1))*t_s**max(0, i-2) for i in range(7, -1, -1)]) @ c)
        x_dddot = (np.array([(i*(i-1)*(i-2))*t_s**max(0, i-3) for i in range(7, -1, -1)]) @ c)
        x_ddddot = (np.array([(i*(i-1)*(i-2)*(i-3))*t_s**max(0, i-4) for i in range(7, -1, -1)]) @ c)
        
        yaw, yaw_dot = 0., 0.
        return x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot


#################### UTILITY FUNCTION ####################
##########################################################


def yield_all_task_trajectories(
    root=DEFAULT_ROOT, 
    dataset_name=DEFAULT_DATASET_NAME, 
    verbose=False
):
    """
    Generator to yield tuple(params, task_name, TrajectoryGenerator)

    Parameters:
        - root: str - The root path for where datasets are located
        - dataset_name: str - The name of the dataset
        - verbose: bool - whether or not to print outputs
    
    Returns:
        - tuple(params, task_name, TrajectoryGenerator)
        
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
            try:
                if params["trajectory_generator"] == "cubic_spline":
                    trajectory_generator = BSplineTrajectoryGenerator(deepcopy(params), f"{root}{dataset_name}/", task_name[:-4])
                elif params["trajectory_generator"] == "min_snap":
                    trajectory_generator = MinSnapTrajectoryGenerator(deepcopy(params), f"{root}{dataset_name}/", task_name[:-4])
                else: #if params["trajectory_generator"] == "constant_speed":
                    trajectory_generator = TrajectoryGenerator(deepcopy(params), f"{root}{dataset_name}/", task_name[:-4])
            except Exception as e:
                trajectory_generator = TrajectoryGenerator(deepcopy(params), f"{root}{dataset_name}/", task_name[:-4])
            yield (
                params,
                task_name[:-4],  
                trajectory_generator
            )


#################### RUNNER ####################
################################################


def run():
    trajectories_by_task = {} 
    prev_task_group = None
    # iterate through all trajectories
    for i, (params, task, traj_gen_obj) in enumerate(yield_all_task_trajectories(verbose=VERBOSE)):
        task_group = "_".join(task.split("_")[1:3])
        # if we have reached a new group of tasks, then
        if task_group not in trajectories_by_task.keys():
            if prev_task_group is not None:
                # for each trajectory in the previous group of tasks,
                for traj in trajectories_by_task[prev_task_group]["generated_trajectories"]:
                    trajectories_by_task[prev_task_group] = collect_task_trajectory(
                                                                trajectories_by_task[prev_task_group], 
                                                                traj
                                                            )
                # Plot Data and free some memory
                if VERBOSE:
                    plot_trajectories_by_task(trajectories_by_task, task_group=prev_task_group)
                trajectories_by_task[prev_task_group]["generated_trajectories"] = None
            
            # then, initialize the next group of tasks
            trajectories_by_task[task_group] = {
                "generated_trajectories": []
            }
        prev_task_group = task_group
        trajectories_by_task[task_group]["generated_trajectories"].append(traj_gen_obj)


if __name__ == "__main__":
    run()
    


