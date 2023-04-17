#################### IMPORTS ####################
#################################################


import os
import numpy as np
import glob
import argparse
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


#################### GLOBAL VARIABLES ####################
##########################################################


DEFAULT_DATA_PATH = "./bootstrap/datasets/AGGRO_000"
DEFAULT_SEED      = 42
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_MAX_RPM   = 21702.6438


#################### PLOTTING UTILITIIES ####################
#############################################################


def plot_dataset(dataset_path, all_ids, all_states, all_control_targets, all_nn_gts, normed=True, sample_inds=None):
    """
    Plot samples from training, validation, or testing data
    """
    split, total_samples = all_ids[0][0], all_ids[-1][-2]+all_ids[-1][-1]
    n_samples = 0
    if split == "train":
        n_samples = 1
    elif split == "val":
        n_samples = 2
    elif split == "test":
        n_samples = 3

    if sample_inds is None:
        trial_length_dist = np.array([all_ids[i][-1] for i in range(len(all_ids))]) / total_samples
        sample_inds = np.random.choice(len(all_ids), size=n_samples, replace=False, p=trial_length_dist)
    
    save_path = os.path.join(dataset_path, "sample_plots")
    os.makedirs(save_path, exist_ok=True)
    for i, s_ind in enumerate(sample_inds):
        s_id = all_ids[s_ind]
        # plot_nn_gts(save_path, s_id, all_nn_gts, normed=normed)
        plot_states(save_path, s_id, all_states[:, :-4], normed=normed)
        plot_states(save_path, s_id, all_control_targets, normed=normed, targets=True)

    return sample_inds


def plot_nn_gts(save_path, s_id, all_nn_gts, normed=False):
    start_ind, end_ind = s_id[-2], s_id[-2]+s_id[-1]
    nn_gts = all_nn_gts[start_ind:end_ind, :].reshape(27, -1, 6)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    plt_ax_map = ["rx", "gy", "bz"]
    trend = torch.mean(nn_gts, dim=0).numpy()
    spread = (trend-0.5*torch.std(nn_gts, dim=0).numpy(), trend+0.5*torch.std(nn_gts, dim=0).numpy())
    # spread = (torch.min(nn_gts, dim=0).values.numpy(), torch.max(nn_gts, dim=0).values.numpy())
    t_trial = np.linspace(0., nn_gts.shape[1]*s_id[-1], num=nn_gts.shape[1])
    for j in range(nn_gts.shape[-1]):
        ax_ind = int(j / 3)
        ax[ax_ind].plot(t_trial, trend[:, j], plt_ax_map[j%3][0], label=f"mean_{plt_ax_map[j%3][1]}")
        ax[ax_ind].fill_between(t_trial, y1=spread[0][:, j], y2=spread[1][:, j], color=plt_ax_map[j%3][0], alpha=0.25) #, label=f"std_{plt_ax_map[j%3][1]}")

    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("a (m/s^2)")
    ax[0].set_title("Kinematic Linear Acceleration")
    ax[0].legend()
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("a (rad/s^2)")
    ax[1].set_title("Kinematic Angular Acceleration")
    ax[1].legend()
    fig.suptitle(f"{'Normalized' if normed else 'Raw'} NN Ground Truth")

    fig_fname = os.path.join(save_path, f"{s_id[0]}_{'normalized' if normed else 'raw'}_nn-gt__{s_id[1][:-4]}.png")
    fig.savefig(fig_fname)
    plt.close()


def plot_states(save_path, s_id, all_states, normed=False, targets=False):
    start_ind, end_ind = s_id[-2], s_id[-2]+s_id[-1]
    states = all_states[start_ind:end_ind, :].reshape(27, -1, 12)

    trend = torch.mean(states, dim=0).numpy()
    spread = (trend-0.5*torch.std(states, dim=0).numpy(), trend+0.5*torch.std(states, dim=0).numpy())
    # spread = (torch.min(nn_gts, dim=0).values.numpy(), torch.max(nn_gts, dim=0).values.numpy())
    t_trial = np.linspace(0., states.shape[1]*s_id[-1], num=states.shape[1])
    for fig_ind in range(2):
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        if fig_ind == 0:
            plt_ax_map = ["rroll", "gpitch", "byaw"]
        else:
            plt_ax_map = ["rx", "gy", "bz"]
        
        j = 6*fig_ind
        for i in range(3):
            ax[0].plot(t_trial, trend[:, j+i], plt_ax_map[i][0], label=f"mean_{plt_ax_map[i][1:]}")
            ax[0].fill_between(t_trial, y1=spread[0][:, j+i], y2=spread[1][:, j+i], color=plt_ax_map[i][0], alpha=0.25)
            ax[1].plot(t_trial, trend[:, j+i+3], plt_ax_map[i][0], label=f"mean_{plt_ax_map[i][1:]}")
            ax[1].fill_between(t_trial, y1=spread[0][:, j+i+3], y2=spread[1][:, j+i+3], color=plt_ax_map[i][0], alpha=0.25)

        ax[0].set_xlabel("t (s)")
        ax[0].set_ylabel("v (m/s)" if fig_ind else "")
        ax[0].set_title("Linear Velocity" if fig_ind else "Sine of Euler Angles")
        ax[0].legend()
        ax[1].set_xlabel("t (s)")
        ax[1].set_ylabel("w (rad/s)" if fig_ind else "")
        ax[1].set_title("Angular Velocity" if fig_ind else "Cosine of Euler Angles")
        ax[1].legend()
        fig.suptitle(f"{'Normalized' if normed else 'Raw'} {'Control Targets' if targets else 'States'}")
        fig_fname = os.path.join(save_path, f"{s_id[0]}_{'normalized' if normed else 'raw'}_{'control-target' if targets else 'state'}_{fig_ind}__{s_id[1][:-4]}.png")
        fig.savefig(fig_fname)
        plt.close()


#################### RUNNER ####################
################################################


def create_torch_dataset(
        dataset_path=DEFAULT_DATA_PATH,
        seed=DEFAULT_SEED,
        val_split=DEFAULT_VAL_SPLIT,
        max_rpm=DEFAULT_MAX_RPM
    ):
    """
    Create a torch dataset for a raw dataset of simulation data
    """

    ##### Find Simulation Data, Create Train-Val Split, and Create Torch Dataset root
    search_path = os.path.join(dataset_path, "sim_data")
    sim_files   = glob.glob(f"{search_path}/*.npy")

    # TODO: Create test datasets/trials for racing type settings so we can use them here 
    train_files, val_files = train_test_split(sim_files, test_size=val_split, random_state=seed)
    val_files, test_files = train_test_split(val_files, test_size=min(len(val_files), 3), random_state=seed)
    files_by_split = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    torch_data_path = os.path.join(dataset_path, "torch_dataset")
    os.makedirs(torch_data_path, exist_ok=True)

    for split, files in files_by_split.items():
        if files is None:
            continue
        
        print(f"Processing {split} data")
        all_ids = []
        all_states = []
        all_control_targets = []
        all_nn_gts = [] 

        ##### Iterate through the files in training, validation, and test
        for trial_fname in tqdm(files):
            
            file_contents = np.load(trial_fname)

            ##### Organize the data from the current flight trial
            states          = file_contents["states"]      # states, size=(num_drones, state_dim, num_samples)
            control_targets = file_contents["controls"]    # desired next states (control targets), size=(num_drones, control_target_dim, num_samples)
            timestamps      = file_contents["timestamps"]  # timestamps, size=(num_drones, num_samples)
            num_drones, state_dim, num_samples = states.shape
            control_target_dim = control_targets.shape[1]

            ##### Reorder the axes of each array (drones, samples, states/control_targets)
            states          = np.transpose(states, axes=(0,2,1))           # transpose operation, size=(num_drones, num_samples, state_dim)
            control_targets = np.transpose(control_targets, axes=(0,2,1))  # transpose operation, size=(num_drones, num_samples, state_dim)

            ##### Before reshaping, compute neural network targets for the neural network: [linear(a_x,a_y,a_z), angular(a_x,a_y,a_z)]
            delta_t = np.diff(timestamps, axis=1)               # size=(num_drones, num_samples-1)
            delta_v = np.diff(states[:, :, 3:6], axis=1)        # size=(num_drones, num_samples-1, 3)
            delta_w = np.diff(states[:, :, 9:12], axis=1)       # size=(num_drones, num_samples-1, 3)
            accel_linear = (delta_v / delta_t[:,:,np.newaxis])  # size=(num_drones, num_samples-1, 3)
            accel_angular = (delta_w / delta_t[:,:,np.newaxis]) # size=(num_drones, num_samples-1, 3)
            nn_gts = np.hstack((accel_linear.reshape(-1, 3), accel_angular.reshape(-1, 3)))  # size=(num_drones*(num_samples-1), 6)
            
            ##### Process the current states: [p_x,p_y,p_z, v_x,v_y,v_z, euler_r,euler_p,euler_y, w_x,w_y,w_z, rpm_0,rpm_1,rpm_2,rpm_3]
            states_no_actions = states[:, :-1, :-4].reshape(-1, state_dim-4) # size=(num_drones*(num_samples-1), state_dim-4)
            actions           = states[:, 1:, -4:].reshape(-1, 4) # indices correspond to the action taken AT index the above states
            rpy = states_no_actions[:, 6:9] # size=(num_drones*(num_samples-1), 3)
            sin_rpy = np.sin(rpy)
            cos_rpy = np.cos(rpy)
            ##### Processed states: [sin(euler_r,euler_p,euler_y), cos(euler_r,euler_p,euler_y), v_x,v_y,v_z, w_x,w_y,w_z, rpm_0,rpm_1,rpm_2,rpm_3]
            states_processed = np.hstack([sin_rpy, cos_rpy, states_no_actions[:,3:6], states_no_actions[:, 9:12], actions])
            # plot_states(states_processed[:, :-4])

            ##### Process the desired states: [p_x,p_y,p_z, euler_r,euler_p,euler_y, v_x,v_y,v_z, w_x,w_y,w_z]
            control_targets = control_targets[:, :-1, :].reshape(-1, control_target_dim) # size=(num_drones*(num_samples-1), control_target_dim)
            rpy = control_targets[:,3:6] # size=(num_drones*(num_samples-1), 3)
            sin_rpy = np.sin(rpy)
            cos_rpy = np.cos(rpy)
            ##### Processed desired states: [sin(euler_r,euler_p,euler_y), cos(euler_r,euler_p,euler_y), v_x,v_y,v_z, w_x,w_y,w_z]
            control_targets_processed = np.hstack([sin_rpy, cos_rpy, control_targets[:,6:]])

            ##### Add the flight trial to the dataset
            all_states.append(torch.tensor(states_processed))
            all_control_targets.append(torch.tensor(control_targets_processed))
            all_nn_gts.append(torch.tensor(nn_gts))

            ##### ID in the dataset (split, trial name, mean_dt, start index, number of samples for the trial)
            if len(all_ids) == 0:
                id = (split, os.path.basename(trial_fname), np.mean(delta_t), 0, num_drones*num_samples)
            else:
                id = (split, os.path.basename(trial_fname), np.mean(delta_t), all_ids[-1][-2] + all_ids[-1][-1], num_drones*num_samples)
            all_ids.append(id)
        
        ##### Combine datasets
        all_states = torch.cat(all_states, dim=0)
        all_control_targets = torch.cat(all_control_targets, dim=0)
        all_nn_gts = torch.cat(all_nn_gts, dim=0)

        ##### Save some sample plots of the raw training, validation, and testing datasets
        sample_inds = plot_dataset(dataset_path, all_ids, all_states, all_control_targets, all_nn_gts, normed=False)

        ##### Min-Max Normalization
        all_states[:,-4:] = all_states[:,-4:] / max_rpm
        state_min          = torch.min(all_states, dim=0).values
        state_max          = torch.max(all_states, dim=0).values
        control_target_min = torch.min(all_control_targets, dim=0).values
        control_target_max = torch.max(all_control_targets, dim=0).values
        nn_gt_min          = torch.min(all_nn_gts, dim=0).values
        nn_gt_max          = torch.max(all_nn_gts, dim=0).values

        control_target_diff = control_target_max - control_target_min
        control_target_diff[control_target_diff == 0] = 1
        all_states = (all_states - state_min)/(state_max - state_min) * 2 - 1
        all_control_targets = (all_control_targets - control_target_min)/control_target_diff * 2 - 1
        all_nn_gts = (all_nn_gts - nn_gt_min)/(nn_gt_max - nn_gt_min) * 2 - 1

        ##### Save corresponding sample plots of the normalized training, validation, and testing datasets
        plot_dataset(dataset_path, all_ids, all_states, all_control_targets, all_nn_gts, sample_inds=sample_inds)

        ##### Save datasets in compressed format
        torch.save(all_states,          os.path.join(torch_data_path, f"{split}_states.pt"))
        torch.save(all_control_targets, os.path.join(torch_data_path, f"{split}_control_targets.pt"))
        torch.save(all_nn_gts,          os.path.join(torch_data_path, f"{split}_nn_gts.pt"))
        ##### Save minimums and maximums in compressed format
        torch.save(state_min,           os.path.join(torch_data_path, f"{split}_state_min.pt"))
        torch.save(control_target_min,  os.path.join(torch_data_path, f"{split}_control_target_min.pt"))
        torch.save(nn_gt_min,           os.path.join(torch_data_path, f"{split}_nn_gt_min.pt"))
        torch.save(state_max,           os.path.join(torch_data_path, f"{split}_state_max.pt"))
        torch.save(control_target_max,  os.path.join(torch_data_path, f"{split}_control_target_max.pt"))
        torch.save(nn_gt_max,           os.path.join(torch_data_path, f"{split}_nn_gt_max.pt"))
    

if __name__ == "__main__":
    ##### Define and parse (optional) arguments for the script
    parser = argparse.ArgumentParser(description='Pytorch dataset file generation script')
    parser.add_argument('--dataset_path', default=DEFAULT_DATA_PATH, type=str, help='Root path of dataset', metavar='')
    parser.add_argument('--seed',         default=DEFAULT_SEED, type=int, help='Seed for dataset splitting', metavar='')
    parser.add_argument('--val_split',    default=DEFAULT_VAL_SPLIT, type=float, help='Fraction of dataset used for validation')
    parser.add_argument('--max_rpm',      default=DEFAULT_MAX_RPM, type=float, help='Maximum RPM value in dataset')
    ARGS = parser.parse_args()

    create_torch_dataset(**vars(ARGS))