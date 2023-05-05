import numpy as np
import glob
import json
import os
import argparse
from scipy.spatial.transform import Rotation

def parse_args():
    parser = argparse.ArgumentParser(description="MPPI Trajectory evaluation script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--traj-dir",
        help="path to directory with trajectory npy files to be evaluated",
        type=str
    )
    args = parser.parse_args()
    return args

def compute_traj_errors(traj_dir):
    traj_files = glob.glob(f"{traj_dir}/*.npy")
    xyz_errors = []
    rpy_errors = []
    for file in traj_files:
        traj = np.load(file)
        tracked_states = traj["states"]
        desired_states = traj["controls"]
        
        # skip empty arrays
        if tracked_states.size == 0:
            continue

        tracked_xyz = tracked_states[0,:3]
        desired_xyz = desired_states[0,:3]
        tracked_rpy = tracked_states[0,6:9]
        desired_rpy = desired_states[0,3:6]

        xyz_error = np.mean(np.sqrt(np.sum((tracked_xyz - desired_xyz)**2, axis=0)))
        tracked_R = Rotation.from_euler('xyz', tracked_rpy.T)
        desired_R = Rotation.from_euler('xyz', desired_rpy.T)
        R_error = tracked_R * desired_R.inv()
        rpy_error = np.mean(R_error.magnitude())

        xyz_errors.append(xyz_error)
        rpy_errors.append(rpy_error)

    xyz_errors = np.array(xyz_errors)
    rpy_errors = np.array(rpy_errors)

    xyz_mean, xyz_std = (np.mean(xyz_errors), np.std(xyz_errors))
    rpy_mean, rpy_std = (np.mean(rpy_errors), np.std(rpy_errors))

    return xyz_mean, xyz_std, rpy_mean, rpy_std

def save_print_errors(traj_dir, errors):
    error_dict = {
        'xyz_error': {'mean': errors[0], 'std': errors[1]},
        'rpy_error': {'mean': errors[2], 'std': errors[3]}
    }

    for k in error_dict.keys():
        print(f"{k}: {error_dict[k]['mean']} +/- {error_dict[k]['std']}")
    
    with open(os.path.join(traj_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(error_dict, f)

if __name__ == "__main__":
    args = parse_args()
    errors = compute_traj_errors(args.traj_dir)
    save_print_errors(args.traj_dir, errors)