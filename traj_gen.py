#################### IMPORTS ####################
#################################################



import os
import csv
from collections import namedtuple
from itertools import product
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.sparse.linalg import lsmr
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from bootstrap.task_battery import DEFAULT_TASK_NAME, DEFAULT_DATASET_NAME, DEFAULT_ROOT
from bootstrap.utils import get_traj_params


#################### TRAJECTORY GENERATORS ####################
###############################################################


class TrajectoryGenerator:
    """
    Base Trajectory Generator Class (Implements Constant Speed Trajectory)
    """
    def __init__(self, 
                 config, 
                 root=DEFAULT_ROOT, 
                 task_name=DEFAULT_TASK_NAME
    ):
        """
        Important variables/dimensions
        self.path           [N, 3]
        self.points         [self.n_wpts, 3]
        self.disp_vecs      [self.n_wpts, 3]
        self.dist_vec       [self.n_wpts,]
        self.unit_vecs      [self.n_wpts, 3]
        self.t_start_vec    [self.n_wpts,]
        self.t_segment_vec  [self.n_wpts - 1,]

        """
        ##### Retrieve the waypoints and config for the current task
        self.root = root
        self.task_name = task_name
        self.config = config
        with open(os.path.join(root, "waypoints", f"{task_name}.csv"), "r", newline="\n") as f:
            self.path = np.vstack(([row for i, row in enumerate(csv.reader(f)) if i > 0])).astype("float")
        self.path -= self.path[0]
        self.downsample_rate = 5  # Path downsampling rate 
        self.is_done = False      # Indicator for task completion 
        # Pre-allocate the desired trajectory dictionary
        self.flat_output = { 'x': None, 'x_dot': None, 'x_ddot': None, 'x_dddot': None, 'x_ddddot': None,
                             'yaw': None, 'yaw_dot': None }
        
        ##### Generate Sparse Waypoints
        rdp_inds = self.filter_waypoints()
        ##### Generate Time Segments
        self.scale_t_segments(rdp_inds=rdp_inds)

    def filter_waypoints(self):
        """
        Filter waypoints and generate displacement, distance, direction vectors.
        Uses RDP Algo for waypoint pruning, and if that fails, linear downsampling.
        Pruned waypoints are then fine-tuned to add fixed interval intermediate points
        to be used as constraints in minimum snap trajectory generation.  
        """
        if "figure_eight" in self.task_name:
            try:
                rdp_points = self.rdp_filtering(self.path)
                points, rdp_inds = self.finetune_waypoints(rdp_points)
            except Exception as e:
                print(e)
                if self.path.shape[0] > 2.*self.downsample_rate:
                    points = self.path[::self.downsample_rate]
                else:
                    points = self.path
                rdp_inds = None
        else:
            if self.path.shape[0] > 2.*self.downsample_rate:
                points = self.path[::self.downsample_rate]
            else:
                points = self.path
            rdp_inds = None

        self.points = points
        self.n_wpts = self.points.shape[0]

        # disp_vects stores the (x, y, z) DISPLACEMENTS between consectutive waypoints, size=(self.n_wpts,3)
        self.disp_vecs = np.zeros((self.n_wpts, 3))
        self.disp_vecs[:-1, :] = np.diff(self.points, n=1, axis=0)
        # dist_vec stores the scalar DISTANCE between consecutive waypoints, size=(self.n_wpts,)
        self.dist_vec = np.linalg.norm(self.disp_vecs, axis=1)
        # unit_vecs stores the (x, y, z) UNIT VECTOR between consecutive waypoints, size=(self.n_wpts, 3)
        self.unit_vecs = np.zeros_like(self.disp_vecs)
        self.unit_vecs[:-1, :] = self.disp_vecs[:-1, :] / self.dist_vec[:-1].reshape(-1, 1)

        return rdp_inds

    def rdp_filtering(self, points):
        """
        Filters waypoints according to ramer-douglas-peucker algorithm
        rdp_eps is the maximum distance measured from the vector along endpoints of a segment
        to a point on that segment.  Points that are outside this range are filtered out
        Generally, 0.1-0.8 are decent choices.

        Parameters:
            - points: np.array(N, 3) - All (x, y, z) points in a dense path
            - rdp_eps: float - Thresholding factor
        
        Returns:
            - filtered_waypoints: np.array(self.n_wpts, 3) - All (x, y, z) points in filtered path

        """
        # Find maximum perpendicular distance from the line formed from the start to the end of this segment
        try:
            max_dist, max_ind = 0, 0
            v = points[-1] - points[0]
            v_norm = np.linalg.norm(v)
            orth_dists = np.linalg.norm(np.cross(points[1:-1] - points[0], v), axis=1) / v_norm
            max_dist, max_ind = np.max(orth_dists), np.argmax(orth_dists)
        except:
            return points
        # If the maximum distance is greater than a threshold, we split the segment in two and recurse
        if (max_dist > self.config.rdp_threshold and points[:max_ind].shape[0] > 2 and points[max_ind:].shape[0] > 2):
            # Recursion 
            return np.vstack((self.rdp_filtering(points[:max_ind])[:-1], self.rdp_filtering(points[max_ind:])))
        else:
            # Base Case: if the maximum distance is less than or equal to the threshold, we keep only the endpoints
            return np.array([points[0], points[-1]])

    def finetune_waypoints(self, rdp_points):
        """
        Adds intermediary points to the already filtered waypoints, to be used as intermediate
        constraints in minimum-snap trajectory generation.  Points are added for each segment 
        greater than 3.5 meters long and add int(length/3.5) + 1 approximately linearly-spaced 
        points from the original UNFILTERED path to the segment.

        Parameters:
            - rdp_points: np.array(N, 3) - All (x, y, z) points in the initially filtered path
        
        Returns:
            - intermediate_points: np.array(self.n_wpts, 3) - the initially filtered path with intermediate points
            - rdp_inds: np.array(N) - the indices of the initially filtered path in intermediate_points

        """
        ##### This adds intermediate waypoints to our rdp pruned path if we prune too aggressively
        intermediate_points = [rdp_points[0].reshape(1, 3)]
        for i, pt_i in enumerate(rdp_points[1:], 1):
            dist = np.linalg.norm(pt_i - rdp_points[i-1])
            num = dist / 3.5 # Add intermediate points for segments longer than 3.5 meters
            if num >= 1: # for segments that are greater than 3 meters long, add intermediate points
                intermediate_points.append(np.vstack([np.linspace(rdp_points[i-1, j], pt_i[j], num=int(num) + 2)[1:] for j in range(3)]).T)
            else:
                intermediate_points.append(pt_i.reshape(1, 3))
        intermediate_points = np.vstack(intermediate_points)
        ##### Snap intermediate points to the original A* Path (potentially unsafe)
        for i, intermediate_pt in enumerate(intermediate_points):
            nearest_path_ind = np.argmin(np.linalg.norm(self.path - intermediate_pt.reshape(1, -1), axis=1))
            intermediate_points[i] = self.path[nearest_path_ind]
        ##### Get the indices of the original rdp filtered waypoints
        rdp_inds = []
        for i, pt_i in enumerate(rdp_points):
            rdp_inds.append(np.argmin(np.sum(np.abs(intermediate_points - pt_i.reshape(1, -1)), axis=1)))
        rdp_inds = np.array(rdp_inds)

        return intermediate_points, rdp_inds

    def scale_t_segments(self, rdp_inds=None):
        """
        Generates start-times for each segment on the path.  Assumes constant speed between waypoints
        to generate time-segments.  First segment and last two segments are scaled by a factor of 0.6
        to account for accelerating from zero velocity and overshooting the goal, respectively.

        Parameters:
            - rdp_inds: np.array(N) - the indices of the initially filtered path in the path with intermediate points
        """
        if rdp_inds is not None:
            rdp_dists = []
            for i, rdp_ind in enumerate(rdp_inds[1:], 1):
                rdp_start = self.points[rdp_inds[i-1]]
                rdp_end = self.points[rdp_ind]
                rdp_dist = np.linalg.norm(rdp_end - rdp_start)
                rdp_dists.extend([rdp_dist for j in range(rdp_inds[i-1], rdp_ind)])
            dists = np.array(rdp_dists)
        else:
            dists = self.dist_vec[:-1]
        dist_norm = np.linalg.norm(dists)
        if dist_norm > 0:
            dists = dists / dist_norm
        ##### Generate speed vector (proportional to segment length) and time-segments assuming constant speed 
        speed_vec = self.config.speed*np.ones_like(dists) + dists
        self.speed_vec = np.append(speed_vec, 0.)
        ##### Speeding up in the first few waypoints
        n_warmup_wpts = 5
        self.speed_vec[:n_warmup_wpts] *= np.arange(1, n_warmup_wpts + 1) / n_warmup_wpts
        self.vel_vecs = self.speed_vec.reshape(-1, 1) * self.unit_vecs
        self.t_start_vec = np.hstack(
            (np.zeros(1), np.cumsum(self.dist_vec[:-1] / self.speed_vec[:-1], axis=0))
        )
        self.t_finish = self.t_start_vec[-1]
        self.t_segment_vec = np.diff(self.t_start_vec) # Define length of time for each segment, hence size=(self.n_wpts-1,)

    def get_trajectory_state(self, t):
        """
        Creates the state tuple for a constant speed trajectory

        Parameters:
            - t: float - the timestamp in seconds

        Returns:
            - (x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot): tuple(np.array(3,) for i in range(5), float, float) - The desired
            state for the current timestamp

        """
        # If we are close to the final waypoint:
        if t+0.25 >= self.t_finish:
            # Set the desired state to be stationary at the final waypoint
            x = self.points[-1]
            x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) for i in range(4))
            self.is_done = True # Task is complete
        else:
            # Otherwise, find the index of the last waypoint that was traversed via t and t_start_vec
            t_diff = self.t_start_vec - t
            last_tstart_ind = np.argwhere(t_diff <= 0)[-1, 0]
            # Compute the relative progress through the segment, and use that to compute the desired position
            t_progress = (t - self.t_start_vec[last_tstart_ind]) / self.t_segment_vec[last_tstart_ind]
            x = (
                self.points[last_tstart_ind]
                + (t_progress * self.dist_vec[last_tstart_ind]) * self.unit_vecs[last_tstart_ind]
            )
            x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) if i > 0 else self.vel_vecs[last_tstart_ind] for i in range(4))

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
            self.flat_output[k] = state[i]
        ##### Return a copy of the flat desired-state dictionary
        return self.flat_output.copy()


#################### B-SPLINE TRAJECTORY GENERATOR ####################
######################################################################


class BSplineTrajectoryGenerator(TrajectoryGenerator):
    """
    Sub-class of TrajectoryGenerator to implement a 5th order interpolating B-Spline
    """
    def __init__(self, 
                 config, 
                 root=DEFAULT_ROOT, 
                 task_name=DEFAULT_TASK_NAME
    ):
        super().__init__(config, root, task_name)
        ##### Create 5th order interpolating spline with boundary conditions
        self.spline_order = 5
        self.spline_path = [
            make_interp_spline(
                self.t_start_vec,
                self.points[:, i],
                k=self.spline_order,
                bc_type=(
                    [(j, 0.0) for j in range(1, 3)], 
                    [(j, 0.0) for j in range(1, 3)])) 
            for i in range(3)
        ]
        
    def get_trajectory_state(self, t):
        """
        Creates the state tuple for a basic k-th order spline interpolation trajectory

        Returns:
            - (x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot): tuple(np.array(3,) for i in range(5), float, float) - The desired
            state for the current timestamp

        """
        # If we are close to the final waypoint:
        if np.any(np.isnan(np.array([self.spline_path[i](t+0.25, extrapolate=False) for i in range(3)]))):
            # Set the desired state to be stationary at the end of the trajectory
            x = self.points[-1]
            x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) for i in range(4))
            self.is_done = True # Task is complete
        else:
            # Otherwise, set the desired state to be that given by the continuous trajectory
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
    Includes a minimum acceleration yaw trajectory
    """
    def __init__(self, 
                 config, 
                 root=DEFAULT_ROOT, 
                 task_name=DEFAULT_TASK_NAME
    ):
        super().__init__(config, root, task_name)

        # Generate global minimum-snap trajectory
        self.minimize_snap()

        # Generate minimum-accel yaw trajectory
        self.minimize_yaw()

        # Debug (visualization)
        # self.simulate_trajectory()

    def minimize_snap(self):
        """
        Minimum Snap Trajectory Optimization using scipy.optimization.lsmr
        Computes an optimal, dynamically feasible trajectory between waypoints using
        continuity and corridor constraints.

        """
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
            t_i, t = self.t_start_vec[i], self.t_segment_vec[i]
            t = t
            # end points
            if i == self.n_wpts - 2:
                A[num_coef - 4:num_coef, num_coef - 8:num_coef] = np.array([
                    [t**7, t**6, t**5, t**4, t**3, t**2, t, 1],
                    [7 * t**6, 6 * t**5, 5 * t**4, 4 * t**3, 3 * t**2, 2 * t, 1, 0],
                    [42 * t**5, 30 * t**4, 20 * t**3, 12 * t**2, 6 * t, 2, 0, 0],
                    [210 * t**4, 120 * t**3, 60 * t**2, 24 * t, 6, 0, 0, 0]])
                b[num_coef - 4] = self.points[i + 1]

            # points between start and end
            else:
                # total 8 rows
                # position constraints x1(t1) = x2(t1) = x1 2*
                A[index, 8 * i: 8 * (i + 1)] = np.array(
                    [[t**7, t**6, t**5, t**4, t**3, t**2, t, 1]])
                A[index + 1, 8 * (i + 1): 8 * (i + 2)] = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])

                b[index] = self.points[i + 1]
                b[index + 1] = self.points[i + 1]

                # continuity constrain x1'(t1) = x2'(t1) 6*
                # velocity
                A[index + 2, 8 * i: 8 * (i + 2)] = np.array(
                    [[7 * t**6, 6 * t**5, 5 * t**4, 4 * t**3, 3 * t**2, 2 * t, 1, 0,
                      0, 0, 0, 0, 0, 0, -1, 0]])

                # acceleration
                A[index + 3, 8 * i: 8 * (i + 2)] = np.array([
                    [42 * t**5, 30 * t**4, 20 * t**3, 12 * t**2, 6 * t, 2, 0, 0,
                     0, 0, 0, 0, 0, -2, 0, 0]])

                # jerk
                A[index + 4, 8 * i: 8 * (i + 2)] = np.array([
                    [210 * t**4, 120 * t**3, 60 * t**2, 24 * t, 6, 0, 0, 0,
                     0, 0, 0, 0, -6, 0, 0, 0]])

                # snap
                A[index + 5, 8 * i: 8 * (i + 2)] = np.array([
                    [840 * t**3, 360 * t**2, 120 * t, 24, 0, 0, 0, 0,
                     0, 0, 0, -24, 0, 0, 0, 0]])

                # crackle
                A[index + 6, 8 * i: 8 * (i + 2)] = np.array([
                    [2520 * t**2, 720 * t, 120, 0, 0, 0, 0, 0,
                     0, 0, -120, 0, 0, 0, 0, 0]])

                # pop
                A[index + 7, 8 * i: 8 * (i + 2)] = np.array([
                    [5040 * t, 720, 0, 0, 0, 0, 0, 0,
                     0, -720, 0, 0, 0, 0, 0, 0]])

                index += 8

        damp, atol, btol, maxiter = 0.0, 1e-9, 1e-9, 10000
        Cx = lsmr(A, b[:, 0], damp=damp, atol=atol, btol=btol, maxiter=maxiter)
        Cy = lsmr(A, b[:, 1], damp=damp, atol=atol, btol=btol, maxiter=maxiter)
        Cz = lsmr(A, b[:, 2], damp=damp, atol=btol, btol=btol, maxiter=maxiter)
        self.c = np.hstack((Cx[0].reshape(len(A), 1), Cy[0].reshape(len(A), 1), Cz[0].reshape(len(A), 1))).reshape(-1, 8, 3)

    def minimize_yaw(self):
        """
        Use minimal acceleration to generate a smooth yaw trajectory
        """
        init_yaw = -np.pi / 2. # This is the initial yaw in the sim
        rpy_points = []
        for i, t_start in enumerate(self.t_start_vec[:-1]):
            ##### Get velocity and accleration at each waypoint
            flat_output = self.update(t_start)
            vel, accel = flat_output["x_dot"], flat_output["x_ddot"]
            ##### Compute Unit Tangent, Unit Normal, and Unit Binormal Vectors
            speed = np.linalg.norm(vel)
            unit_tangent = vel / speed
            tan_accel = np.dot(unit_tangent, accel)
            normal_accel = accel - tan_accel * unit_tangent
            unit_normal = normal_accel / np.linalg.norm(normal_accel)
            unit_binormal = np.cross(unit_tangent, unit_normal)
            ##### Construct rotation matrix from unit vectors, store the euler angles with 0 roll, 0 pitch
            rot = Rotation.from_matrix(np.vstack((unit_tangent, unit_normal, unit_binormal)).T)
            yaw_rot = rot.as_euler("xyz")[-1]
            rpy_points.append([0., 0., init_yaw + yaw_rot])
        
        ##### Obtain final yaw
        rpy_points = np.array(rpy_points)
        quadrants = np.array([0., np.pi / 2., np.pi, 3*np.pi / 2., 2*np.pi])
        final_yaw = quadrants[np.argmin(np.abs(quadrants - rpy_points[-1, 2]))]
        rpy_points = np.vstack(([0., 0., init_yaw], rpy_points[1:], [0., 0., final_yaw]))

        ##### Prune yaw waypoints where delta is small
        self.yaw_t_start_inds = np.flatnonzero(np.abs(np.diff(rpy_points, n=1, axis=0))[:, -1] > (10*np.pi / 180))
        if rpy_points.shape[0] - 1 not in self.yaw_t_start_inds:
            self.yaw_t_start_inds = np.append(self.yaw_t_start_inds, rpy_points.shape[0] - 1)
        if 0 not in self.yaw_t_start_inds:
            self.yaw_t_start_inds = np.insert(self.yaw_t_start_inds, 0, 0)
        if self.yaw_t_start_inds.shape[0] > 2:
            rpy_points = rpy_points[self.yaw_t_start_inds, :]
            yaw_t_start_vec = self.t_start_vec[self.yaw_t_start_inds]
        else:
            yaw_t_start_vec = self.t_start_vec
        
        ##### Create yaw waypoints, smooth them, and create a Cubic interpolating spline between smoothed waypoints
        sg_window = min(5, rpy_points.shape[0] - 1)
        sg_order = min(3, sg_window - 1)
        rpy_points = savgol_filter(rpy_points, sg_window, sg_order, axis=0)
        self.rot_points_obj = Rotation.from_euler("xyz", rpy_points)
        self.yaw_generator = RotationSpline(yaw_t_start_vec, self.rot_points_obj)

    def simulate_trajectory(self):
        """
        Simulate the trajectory, for debugging purposes
        """
        ##### Rollout the entire global trajectory
        flat_output = self.update(0.)
        yaw_sim = [self.yaw_generator(0., 0)]
        yaw_dot_sim = [self.yaw_generator(0., 1)]
        xyz_sim = flat_output["x"]
        T = np.arange(0, self.t_finish, 0.01)
        if self.yaw_t_start_inds.shape[0] > 2:
            t_start_inds = self.yaw_t_start_inds # indexes for yaw waypoints
        else:
            t_start_inds = np.arange(self.t_start_vec.shape[0]) # timesteps
        for t in T[1:]:
            flat_output = self.update(t)
            yaw_sim.append(self.yaw_generator(t, 0))
            yaw_dot_sim.append(self.yaw_generator(t, 1))
            xyz_sim = np.vstack((xyz_sim, flat_output["x"]))
        yaw_sim = Rotation.concatenate(yaw_sim)
        yaw_dot_sim = np.array(yaw_dot_sim)

        ##### Plot RotationSpline euler angles
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(T, yaw_sim.as_euler("xyz")[:, 0], "r-")
        ax[1].plot(T, yaw_sim.as_euler("xyz")[:, 1], "g-")
        ax[2].plot(T, np.unwrap(yaw_sim.as_euler("xyz")[:, 2]), "b-", label="yaw spline")
        ##### Plot yaw keypoints in 2D
        ax[0].scatter(self.t_start_vec[t_start_inds], self.rot_points_obj.as_euler("xyz")[:, 0], c="r", label="waypoint")
        ax[1].scatter(self.t_start_vec[t_start_inds], self.rot_points_obj.as_euler("xyz")[:, 1], c="g", label="waypoint")
        ax[2].scatter(self.t_start_vec[t_start_inds], np.unwrap(self.rot_points_obj.as_euler("xyz")[:, 2]), c="b", label="waypoint")

        ax[0].set_ylabel("roll")
        ax[1].set_ylabel("pitch")
        ax[2].set_ylabel("yaw")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        plt.show()

        ###### Plot the X-Y global trajectory
        fig = plt.figure(2)
        ax = fig.add_subplot()
        ax.plot(*xyz_sim[:, :-1].T)
        ###### Plot the Yaw keypoints
        x_directions = self.rot_points_obj.apply(np.array([0.1, 0., 0.]))
        y_directions = self.rot_points_obj.apply(np.array([0., 0.1, 0.]))
        lines, all_axes = [], []
        for i, t in enumerate(t_start_inds):
            x_axis = np.vstack((self.points[t] - 0.5*x_directions[i], self.points[t], self.points[t] + x_directions[i])).T
            y_axis = np.vstack((self.points[t] - 0.5*y_directions[i], self.points[t], self.points[t] + y_directions[i])).T
            lines, = ax.plot(*x_axis[:-1, :], "r-")
            lines, = ax.plot(*y_axis[:-1, :], "g-")
            all_axes.append(lines)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.show()

        ##### Plot the 3D global trajectory
        fig = plt.figure(3)
        ax = fig.add_subplot(projection="3d")
        ax.plot(*xyz_sim.T)
        ##### Animate the RotationSpline
        x_directions = yaw_sim.apply(np.array([0.1, 0.,  0.]))
        y_directions = yaw_sim.apply(np.array([0.,  0.1, 0.]))
        z_directions = np.cross(y_directions, x_directions)
        lines = []
        lines.extend(ax.plot(*np.vstack((xyz_sim[0] - 0.5*x_directions[0], xyz_sim[0], xyz_sim[0] + x_directions[0])).T, "r-", label="x"))
        lines.extend(ax.plot(*np.vstack((xyz_sim[0] - 0.5*y_directions[0], xyz_sim[0], xyz_sim[0] + y_directions[0])).T, "g-", label="y"))
        lines.extend(ax.plot(*np.vstack((xyz_sim[0] - 0.5*z_directions[0], xyz_sim[0], xyz_sim[0] + z_directions[0])).T, "b-", label="z"))
        ##### Simple Animation Function
        def ani(t):
            lines[0].set_data_3d(*np.vstack((xyz_sim[t] - 0.5*x_directions[t], xyz_sim[t], xyz_sim[t] + x_directions[t])).T)
            lines[1].set_data_3d(*np.vstack((xyz_sim[t] - 0.5*y_directions[t], xyz_sim[t], xyz_sim[t] + y_directions[t])).T)
            lines[2].set_data_3d(*np.vstack((xyz_sim[t] - 0.5*z_directions[t], xyz_sim[t], xyz_sim[t] + z_directions[t])).T)
            return lines

        lims = (np.min(xyz_sim), np.max(xyz_sim))
        ax.set_xlim3d(*lims)
        ax.set_ylim3d(*lims)
        ax.set_zlim3d(*lims)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        animator = FuncAnimation(fig, ani, list(range(0, T.shape[0], 5)), interval=50)
        plt.show()


    def get_trajectory_state(self, t):
        """
        Creates the state tuple for a minimum snap trajectory

        Returns:
            - (x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot): tuple(np.array(3,) for i in range(5), float, float) - The desired
            state for the current timestamp

        """
        t_diff = self.t_start_vec - t
        last_tstart_ind = np.argwhere(t_diff <= 0)[-1, 0]
        if t+0.25 >= self.t_finish:
            x = self.points[-1]
            x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) for i in range(4))
            yaw, yaw_dot = self.rot_points_obj.as_euler("xyz", degrees=False)[-1, -1], 0.
            self.is_done = True
        else:
            t_s = float(t - self.t_start_vec[last_tstart_ind])
            c = self.c[last_tstart_ind]
            if t == 0:
                x = self.points[0]
            else:
                x = (np.array([t_s**i for i in range(7, -1, -1)]) @ c)
            x_dot = (np.array([float(i)*t_s**max(0, i-1) for i in range(7, -1, -1)]) @ c)
            x_ddot = (np.array([float(i*(i-1))*t_s**max(0, i-2) for i in range(7, -1, -1)]) @ c)
            x_dddot = (np.array([float(i*(i-1)*(i-2))*t_s**max(0, i-3) for i in range(7, -1, -1)]) @ c)
            x_ddddot = (np.array([float(i*(i-1)*(i-2)*(i-3))*t_s**max(0, i-4) for i in range(7, -1, -1)]) @ c)
            
            try:
                yaw, yaw_dot = self.yaw_generator(t, 0).as_euler("xyz")[-1], self.yaw_generator(t, 1)[-1]
            except:
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
    Generator to yield tuple(config, task_name, TrajectoryGenerator)

    Parameters:
        - root: str - The root path for where datasets are located
        - dataset_name: str - The name of the dataset
        - verbose: bool - whether or not to print outputs
    
    Returns:
        - tuple(config, task_name, TrajectoryGenerator)
        
    """
    dataset_dir = os.path.join(root, dataset_name)
    if verbose: 
        tasks = tqdm(sorted(os.listdir(os.path.join(dataset_dir, "waypoints"))))
    else:
        tasks = sorted(os.listdir(os.path.join(dataset_dir, "waypoints")))
    for task_name in tasks:
        # get the parameters for the current task
        config_dict = get_traj_params(root, dataset_name, task_name[:-4])
        # for parameters that we must control at trajectory generation time,
        # we create a grid in the same way that we do in task_gen.py
        traj_search_params = []
        for k, v in sorted(config_dict.items(), key=lambda x: x[0]):
            if type(v) == list:
                traj_search_params.append(k)
        traj_param_grid = product(*[config_dict[k] for k in traj_search_params])
        # iterate through the trajectory specific parameter grid
        for i, traj_task_params in enumerate(traj_param_grid):
            # set the traj_search_params to their values in the current grid-search
            for j, key in enumerate(traj_search_params):
                config_dict[key] = traj_task_params[j]
            if verbose:
                # display progress
                print(f"Creating Trajectory Object for {task_name[:-4]}", end="\r", flush=True)
            config = namedtuple("trajectory_config", config_dict.keys())(**config_dict)
            # yield the current config, task_name, and the corresponding Trajectory Generator Object
            try:
                if config_dict["trajectory_generator"] == "cubic_spline":
                    trajectory_generator = BSplineTrajectoryGenerator(config, dataset_dir, task_name[:-4])
                elif config_dict["trajectory_generator"] == "min_snap":
                    trajectory_generator = MinSnapTrajectoryGenerator(config, dataset_dir, task_name[:-4])
                else:
                    trajectory_generator = TrajectoryGenerator(config, dataset_dir, task_name[:-4])
            except Exception as e:
                trajectory_generator = TrajectoryGenerator(config, dataset_dir, task_name[:-4])
            yield (
                config,
                task_name[:-4],  
                trajectory_generator
            )


# #################### IMPORTS ####################
# #################################################


# import os
# import csv
# from tqdm import tqdm
# from collections import namedtuple
# from itertools import product
# import numpy as np
# from scipy.interpolate import make_interp_spline
# from scipy.sparse.linalg import lsmr

# from bootstrap.task_battery import DEFAULT_TASK_NAME, DEFAULT_DATASET_NAME, DEFAULT_ROOT
# from bootstrap.utils import get_traj_params


# #################### TRAJECTORY GENERATORS ####################
# ###############################################################


# class TrajectoryGenerator:
#     """
#     Base Trajectory Generator Class (Implements Constant Speed Trajectory)
#     """
#     def __init__(self, 
#                  config, 
#                  root=DEFAULT_ROOT, 
#                  task_name=DEFAULT_TASK_NAME
#     ):
#         """
#         Important variables/dimensions
#         self.path           [N, 3]
#         self.points         [self.n_wpts, 3]
#         self.disp_vecs      [self.n_wpts, 3]
#         self.dist_vec       [self.n_wpts,]
#         self.unit_vecs      [self.n_wpts, 3]
#         self.t_start_vec    [self.n_wpts,]
#         self.t_segment_vec  [self.n_wpts - 1,]
#         """
#         self.root = root
#         self.task_name = task_name
#         self.config = config
#         with open(os.path.join(root, "waypoints", f"{task_name}.csv"), "r", newline="\n") as f:
#             self.path = np.vstack(([row for i, row in enumerate(csv.reader(f)) if i > 0])).astype("float")

#         # Generate Sparse Waypoints (only for complex paths)
#         self.filter_waypoints()

#         # Create vector of start times of each waypoint, assuming constant speed trajectory
#         self.vel_vecs = self.config.speed * self.unit_vecs
#         self.speed_vec = np.linalg.norm(self.vel_vecs, axis=1)
#         self.t_start_vec = np.hstack(
#             (np.zeros(1), np.cumsum( self.dist_vec[:-1] / self.config.speed, axis=0)) # self.dist_vec[:-1] / self.speed_vec[:-1], axis=0))
#         )
#         self.t_finish = self.t_start_vec[-1]
#         self.t_segment_vec = np.diff(self.t_start_vec) # Define length of time for each segment, hence size=(self.n_wpts-1,)

#         self.is_done = False # Indicator for task completion 

#         # Pre-allocate the desired trajectory dictionary
#         self.flat_output = { 'x': None, 'x_dot': None, 'x_ddot': None, 'x_dddot': None, 'x_ddddot': None,
#                              'yaw': None, 'yaw_dot': None }


#     def filter_waypoints(self):
#         """
#         Filter waypoints by:
#             1. Ramer Douglas Peucker Algorithm
#             2. linear downsampling
#         """

#         if "figure_eight" in self.task_name:
#             self.points = self.rdp_filtering(self.path)
#         else:
#             self.points = np.vstack((self.path[0, :], self.path[1:-1:5, :], self.path[-1, :]))
#         self.points = np.around(self.points, 10) # round the precision of waypoints for quality of life
#         self.points -= self.points[0] # normalize waypoints to the first point
#         self.n_wpts = self.points.shape[0]

#         # disp_vects stores the (x, y, z) DISPLACEMENTS between consectutive waypoints, size=(self.n_wpts,3)
#         self.disp_vecs = np.zeros((self.n_wpts, 3))
#         self.disp_vecs[:-1, :] = np.diff(self.points, n=1, axis=0)
#         # dist_vec stores the scalar DISTANCE between consecutive waypoints, size=(self.n_wpts,)
#         self.dist_vec = np.linalg.norm(self.disp_vecs, axis=1)
#         # unit_vecs stores the (x, y, z) UNIT VECTOR between consecutive waypoints, size=(self.n_wpts, 3)
#         self.unit_vecs = np.zeros_like(self.disp_vecs)
#         self.unit_vecs[:-1, :] = self.disp_vecs[:-1, :] / self.dist_vec[:-1].reshape(-1, 1)


#     def rdp_filtering(self, points):
#         """
#         Filters waypoints according to ramer-douglas-peucker algorithm

#         Parameters:
#             - points: np.array(N, 3) - All (x, y, z) points in a dense path
        
#         Returns:
#             - filtered_waypoints: np.array(self.n_wpts, 3) - All (x, y, z) points in filtered path

#         """
#         # Find maximum perpendicular distance from the line formed from the start to the end of this segment
#         max_dist, max_ind = 0, 0
#         v = points[-1] - points[0]
#         v_norm = np.linalg.norm(v)
#         orth_dists = np.linalg.norm(np.cross(points[1:-1] - points[0], v), axis=1) / v_norm
#         max_dist, max_ind = np.max(orth_dists), np.argmax(orth_dists)
        
#         # If the maximum distance is greater than a threshold, we split the segment in two and recurse
#         if (max_dist > self.config.rdp_threshold):
#             # Recursion 
#             return np.vstack((self.rdp_filtering(points[:max_ind])[:-1], self.rdp_filtering(points[max_ind:])))
#         else:
#             # Base Case: if the maximum distance is less than or equal to the threshold, we keep only the endpoints
#             return np.array([points[0], points[-1]])


#     def get_trajectory_state(self, t):
#         """
#         Creates the state tuple for a constant speed trajectory

#         Parameters:
#             - t: float - the timestamp in seconds

#         Returns:
#             - (x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot): tuple(np.array(3,) for i in range(5), float, float) - The desired
#             state for the current timestamp

#         """
#         # If we are close to the final waypoint:
#         if t+0.25 >= self.t_finish:
#             # Set the desired state to be stationary at the final waypoint
#             x = self.points[-1]
#             x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) for i in range(4))
#             self.is_done = True # Task is complete
#         else:
#             # Otherwise, find the index of the last waypoint that was traversed via t and t_start_vec
#             t_diff = self.t_start_vec - t
#             last_tstart_ind = np.argwhere(t_diff <= 0)[-1, 0]
#             # Compute the relative progress through the segment, and use that to compute the desired position
#             t_progress = (t - self.t_start_vec[last_tstart_ind]) / self.t_segment_vec[last_tstart_ind]
#             x = (
#                 self.points[last_tstart_ind]
#                 + (t_progress * self.dist_vec[last_tstart_ind]) * self.unit_vecs[last_tstart_ind]
#             )
#             x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) if i > 0 else self.vel_vecs[last_tstart_ind] for i in range(4))

#         yaw, yaw_dot = 0.0, 0.0

#         return x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot


#     def update(self, t):
#         """
#         Given the present time, return the desired flat output and derivatives.

#         Parameters:
#             - t: float - timestamp in seconds
        
#         Returns:
#             - flat_output: dict() - a dict describing the present desired flat outputs with keys
#                 x,        position, m
#                 x_dot,    velocity, m/s
#                 x_ddot,   acceleration, m/s**2
#                 x_dddot,  jerk, m/s**3
#                 x_ddddot, snap, m/s**4
#                 yaw,      yaw angle, rad
#                 yaw_dot,  yaw rate, rad/s

#         """
#         state = self.get_trajectory_state(t)

#         for i, k in enumerate(self.flat_output.keys()):
#             self.flat_output[k] = state[i]

#         return self.flat_output.copy()


# #################### B-SPLINE TRAJECTORY GENERATOR ####################
# ######################################################################


# class BSplineTrajectoryGenerator(TrajectoryGenerator):
#     """
#     Sub-class of TrajectoryGenerator to implement a k-th order interpolating B-Spline
#     """
#     def __init__(self, 
#                  config, 
#                  root=DEFAULT_ROOT, 
#                  task_name=DEFAULT_TASK_NAME
#     ):
#         super().__init__(config, root, task_name)
#         self.spline_order = 5
#         self.spline_path = [
#             make_interp_spline(
#                 self.t_start_vec,
#                 self.points[:, i],
#                 k=self.spline_order,
#                 bc_type=(
#                     [(j, 0.0) for j in range(1, 3)], 
#                     [(j, 0.0) for j in range(1, 3)])) 
#             for i in range(3)
#         ]
        
    
#     def get_trajectory_state(self, t):
#         """
#         Creates the state tuple for a basic k-th order spline interpolation trajectory

#         Returns:
#             - (x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot): tuple(np.array(3,) for i in range(5), float, float) - The desired
#             state for the current timestamp

#         """
#         # If we are close to the final waypoint:
#         if np.any(np.isnan(np.array([self.spline_path[i](t+0.25, extrapolate=False) for i in range(3)]))):
#             # Set the desired state to be stationary at the end of the trajectory
#             x = self.points[-1]
#             x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) for i in range(4))
#             self.is_done = True # Task is complete
#         else:
#             # Otherwise, set the desired state to be that given by the continuous trajectory
#             x, x_dot, x_ddot, x_dddot, x_ddddot  = (
#                 np.array([
#                     self.spline_path[i](t, nu=nu, extrapolate=False) 
#                     for i in range(3)
#                 ]) for nu in range(self.spline_order)
#             )
        
#         yaw, yaw_dot = 0.0, 0.0
        
#         return x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot
        

# #################### MINIMUM SNAP TRAJECTORY GENERATOR ####################
# ###########################################################################


# class MinSnapTrajectoryGenerator(TrajectoryGenerator):
#     """
#     Sub-class of TrajectoryGenerator to implement minimum snap trajectory generation
#     """
#     def __init__(self, 
#                  config, 
#                  root=DEFAULT_ROOT, 
#                  task_name=DEFAULT_TASK_NAME
#     ):
#         super().__init__(config, root, task_name)

#         # minimum snap
#         num_coef = 8 * (self.n_wpts - 1)  # unknown c to solve for every segments
#         A = np.zeros((num_coef, num_coef))
#         b = np.zeros((num_coef, 3))
#         index = 0  # keep records of rows

#         # start
#         A[0:4, 0:8] = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
#                                 [0, 0, 0, 0, 0, 0, 1, 0],
#                                 [0, 0, 0, 0, 0, 2, 0, 0],
#                                 [0, 0, 0, 0, 6, 0, 0, 0]])
#         b[0, :] = self.points[0]
#         index += 4

#         # update the matrix for each segment
#         for i in range(self.n_wpts - 1):
#             t_i, t = self.t_start_vec[i], self.t_segment_vec[i]
#             # end points
#             if i == self.n_wpts - 2:
#                 A[num_coef - 4:num_coef, num_coef - 8:num_coef] = np.array([
#                     [t**7, t**6, t**5, t**4, t**3, t**2, t, 1],
#                     [7 * t**6, 6 * t**5, 5 * t**4, 4 * t**3, 3 * t**2, 2 * t, 1, 0],
#                     [42 * t**5, 30 * t**4, 20 * t**3, 12 * t**2, 6 * t, 2, 0, 0],
#                     [210 * t**4, 120 * t**3, 60 * t**2, 24 * t, 6, 0, 0, 0]])
#                 b[num_coef - 4] = self.points[i + 1]

#             # points between start and end
#             else:
#                 # total 8 rows
#                 # position constraints x1(t1) = x2(t1) = x1 2*
#                 A[index, 8 * i: 8 * (i + 1)] = np.array(
#                     [[t**7, t**6, t**5, t**4, t**3, t**2, t, 1]])
#                 A[index + 1, 8 * (i + 1): 8 * (i + 2)] = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])

#                 b[index] = self.points[i + 1]
#                 b[index + 1] = self.points[i + 1]

#                 # continuity constrain x1'(t1) = x2'(t1) 6*
#                 # velocity
#                 A[index + 2, 8 * i: 8 * (i + 2)] = np.array(
#                     [[7 * t**6, 6 * t**5, 5 * t**4, 4 * t**3, 3 * t**2, 2 * t, 1, 0,
#                       0, 0, 0, 0, 0, 0, -1, 0]])

#                 # acceleration
#                 A[index + 3, 8 * i: 8 * (i + 2)] = np.array([
#                     [42 * t**5, 30 * t**4, 20 * t**3, 12 * t**2, 6 * t, 2, 0, 0,
#                      0, 0, 0, 0, 0, -2, 0, 0]])

#                 # jerk
#                 A[index + 4, 8 * i: 8 * (i + 2)] = np.array([
#                     [210 * t**4, 120 * t**3, 60 * t**2, 24 * t, 6, 0, 0, 0,
#                      0, 0, 0, 0, -6, 0, 0, 0]])

#                 # snap
#                 A[index + 5, 8 * i: 8 * (i + 2)] = np.array([
#                     [840 * t**3, 360 * t**2, 120 * t, 24, 0, 0, 0, 0,
#                      0, 0, 0, -24, 0, 0, 0, 0]])

#                 # crackle
#                 A[index + 6, 8 * i: 8 * (i + 2)] = np.array([
#                     [2520 * t**2, 720 * t, 120, 0, 0, 0, 0, 0,
#                      0, 0, -120, 0, 0, 0, 0, 0]])

#                 # pop
#                 A[index + 7, 8 * i: 8 * (i + 2)] = np.array([
#                     [5040 * t, 720, 0, 0, 0, 0, 0, 0,
#                      0, -720, 0, 0, 0, 0, 0, 0]])

#                 index += 8

#         damp, atol, btol, maxiter = 0.0, 1e-8, 1e-8, 5000
#         Cx = lsmr(A, b[:, 0], damp=damp, atol=atol, btol=btol, maxiter=maxiter)
#         Cy = lsmr(A, b[:, 1], damp=damp, atol=atol, btol=btol, maxiter=maxiter)
#         Cz = lsmr(A, b[:, 2], damp=damp, atol=btol, btol=btol, maxiter=maxiter)
#         self.c = np.hstack((Cx[0].reshape(len(A), 1), Cy[0].reshape(len(A), 1), Cz[0].reshape(len(A), 1))).reshape(-1, 8, 3)

        
#     def get_trajectory_state(self, t):
#         """
#         Creates the state tuple for a minimum snap trajectory

#         Returns:
#             - (x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot): tuple(np.array(3,) for i in range(5), float, float) - The desired
#             state for the current timestamp

#         """
#         t_diff = self.t_start_vec - t
#         last_tstart_ind = np.argwhere(t_diff <= 0)[-1, 0]
#         if t+0.25 >= self.t_finish:
#             x = self.points[-1]
#             x_dot, x_ddot, x_dddot, x_ddddot = (np.zeros(3) for i in range(4))
#             self.is_done = True
#         else:
#             t_s = float(t - self.t_start_vec[last_tstart_ind])
#             c = self.c[last_tstart_ind]
#             if t == 0:
#                 x = self.points[0]
#             else:
#                 x = (np.array([t_s**i for i in range(7, -1, -1)]) @ c)
#             x_dot = (np.array([float(i)*t_s**max(0, i-1) for i in range(7, -1, -1)]) @ c)
#             x_ddot = (np.array([float(i*(i-1))*t_s**max(0, i-2) for i in range(7, -1, -1)]) @ c)
#             x_dddot = (np.array([float(i*(i-1)*(i-2))*t_s**max(0, i-3) for i in range(7, -1, -1)]) @ c)
#             x_ddddot = (np.array([float(i*(i-1)*(i-2)*(i-3))*t_s**max(0, i-4) for i in range(7, -1, -1)]) @ c)
        
#         yaw, yaw_dot = 0., 0.
#         return x, x_dot, x_ddot, x_dddot, x_ddddot, yaw, yaw_dot