#################### IMPORTS ####################
#################################################

import os
import numpy as np
import glob
import argparse
import torch
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation


#################### GLOBAL VARIABLES ####################
##########################################################


DEFAULT_DATA_PATH = "bootstrap/datasets/AGGRO_000"
DEFAULT_SEED = 42
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_MAX_RPM = 21702.6438


#################### RUNNER ####################
################################################


def run(
        dataset_path=DEFAULT_DATA_PATH,
        seed=DEFAULT_SEED,
        val_split=DEFAULT_VAL_SPLIT,
        max_rpm=DEFAULT_MAX_RPM
    ):
    search_path = os.path.join(dataset_path, "sim_data")
    sim_files = glob.glob(f"{search_path}/*.npy")
    train_files, val_files = train_test_split(sim_files, test_size=val_split, random_state=seed)
    files_by_split = {
        'train': train_files,
        'val': val_files
    }
    torch_data_path = os.path.join(dataset_path, "torch_dataset")
    os.makedirs(torch_data_path, exist_ok=True)

    for split in files_by_split.keys():

        all_states = []
        all_actions = []
        all_targets = []

        for i, f in enumerate(files_by_split[split]):
            print(f"Processing file {i+1} of {len(sim_files)}")
            file_contents = np.load(f)
            states = file_contents["states"][:,:,:-1]
            future_states = file_contents["states"][:,:,1:]
            actions = file_contents["controls"]
            timestamps = file_contents["timestamps"][:,:-1]
            future_timestamps = file_contents["timestamps"][:,1:]
            state_dim = states.shape[1]
            action_dim = actions.shape[1]
            states = np.transpose(states, axes=(0,2,1)).reshape(-1, state_dim)
            future_states = np.transpose(future_states, axes=(0,2,1)).reshape(-1, state_dim)
            actions = np.transpose(actions, axes=(0,2,1)).reshape(-1, action_dim)
            timestamps = timestamps.reshape(-1)
            future_timestamps = future_timestamps.reshape(-1)

            ### PROCESS STATES ###
            rpy = states[:,6:9]
            sin_rpy = np.sin(rpy)
            cos_rpy = np.cos(rpy)
            # excluding position
            states_proc = np.concatenate([sin_rpy, cos_rpy, states[:,3:6], states[:, 9:]], axis=1)

            ### PROCESS ACTIONS ###
            rpy = actions[:,6:9]
            sin_rpy = np.sin(rpy)
            cos_rpy = np.cos(rpy)
            actions_proc = np.concatenate([sin_rpy, cos_rpy, actions[:,3:6], actions[:, 9:]], axis=1)

            ### COMPUTE TARGETS ###
            dt = future_timestamps - timestamps
            lin_targets = (future_states[:,3:6] - states[:,3:6]) / np.expand_dims(dt, axis=1)
            rot_targets = (future_states[:,3:6] - states[:,3:6]) / np.expand_dims(dt, axis=1)
            targets = np.hstack((lin_targets, rot_targets))

            all_states.append(torch.tensor(states_proc))
            all_actions.append(torch.tensor(actions_proc))
            all_targets.append(torch.tensor(targets))
        
        all_states = torch.cat(all_states, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

       ### MIN-MAX NORMALIZATION
        all_states[:,-4:] = all_states[:,-4:] / max_rpm
        state_min = torch.min(all_states, dim=0).values
        state_max = torch.max(all_states, dim=0).values
        action_min = torch.min(all_actions, dim=0).values
        action_max = torch.max(all_actions, dim=0).values
        target_min = torch.min(all_targets, dim=0).values
        target_max = torch.max(all_targets, dim=0).values
        action_diff = action_max - action_min
        action_diff[action_diff == 0] = 1
        all_states = (all_states - state_min)/(state_max - state_min) * 2 - 1
        all_actions = (all_actions - action_min)/action_diff * 2 - 1
        all_targets = (all_targets - target_min)/(target_max - target_min) * 2 - 1

        torch.save(all_states, os.path.join(torch_data_path, f"{split}_states.pt"))
        torch.save(all_actions, os.path.join(torch_data_path, f"{split}_actions.pt"))
        torch.save(all_targets, os.path.join(torch_data_path, f"{split}_targets.pt"))
        torch.save(state_min, os.path.join(torch_data_path, f"{split}_states_min.pt"))
        torch.save(action_min, os.path.join(torch_data_path, f"{split}_actions_min.pt"))
        torch.save(target_min, os.path.join(torch_data_path, f"{split}_targets_min.pt"))
        torch.save(state_max, os.path.join(torch_data_path, f"{split}_states_max.pt"))
        torch.save(action_max, os.path.join(torch_data_path, f"{split}_actions_max.pt"))
        torch.save(target_max, os.path.join(torch_data_path, f"{split}_targets_max.pt"))
    

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Pytorch dataset file generation script')
    parser.add_argument('--dataset_path', default=DEFAULT_DATA_PATH, type=str, help='Root path of dataset', metavar='')
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int, help='Seed for dataset splitting', metavar='')
    parser.add_argument('--val_split', default=DEFAULT_VAL_SPLIT, type=float, help='Fraction of dataset used for validation')
    parser.add_argument('--max_rpm', default=DEFAULT_MAX_RPM, type=float, help='Maximum RPM value in dataset')
    ARGS = parser.parse_args()

    run(**vars(ARGS))