from matplotlib import pyplot as plt
import numpy as np
import glob
import os


DATASET_ROOT = "./bootstrap/datasets/"
DATASETS = [os.path.join(DATASET_ROOT, "mppi"), os.path.join(DATASET_ROOT, "pid")]

def plot_summary_data():
    eval_data_by_controller = {}
    for dataset in DATASETS:
        eval_files = glob.glob(f"{dataset}/**/sim_data/*_eval_data.npy", recursive=True)
        ##### Even inds will be the figure eight task, Odd inds will be the straightaway task
        eval_files = sorted(eval_files, key=lambda x: x[len(dataset) + 1:])
        if "mppi" in dataset:
            eval_key = "norm_rollout_error"
            analytical_eval_data = []
            neural_eval_data = []
            for eval_file in eval_files:
                if "TEST_000" in eval_file:
                    analytical_eval_data.append(np.load(eval_file)[eval_key])
                elif "TEST_001" in eval_file:
                    neural_eval_data.append(np.load(eval_file)[eval_key])
            eval_data_by_controller["analytical_mppi"] = analytical_eval_data
            eval_data_by_controller["neural_mppi"] = neural_eval_data
        else:
            eval_key = "norm_tracking_error"
            pid_eval_data = []
            for eval_file in eval_files:
                pid_eval_data.append(np.load(eval_file)[eval_key])
            eval_data_by_controller["pid"] = pid_eval_data

    for i, physics_model in enumerate(["dyn", "pyb", "drag"]):
        cols = 2
        rows = 4
        fig, ax = plt.subplots(4, 2, figsize=(10, 7))
        controller_colors = ["r", "g", "b"]
        row_mapping = ["p", "v", "q", "w"]
        for j, (controller, controller_eval_data) in enumerate(eval_data_by_controller.items()):
            for row in range(rows):
                ylabel = f"{row_mapping[row]} error"
                for col in range(cols):
                    try:
                        mean_error = np.mean(controller_eval_data[i*cols+col][:, row, :], axis=0)
                        std_error = np.std(controller_eval_data[i*cols+col][:, row, :], axis=0)
                        ax[row, col].plot(np.arange(mean_error.shape[0]) / 48., mean_error, c=controller_colors[j], label=controller)
                        ax[row, col].fill_between(np.arange(mean_error.shape[0]) / 48., mean_error - std_error, mean_error + std_error, color=controller_colors[j], alpha=0.25)
                        ax[row, col].set_ylabel(ylabel)
                        ax[row, col].legend(loc="upper left")
                    except Exception as e:
                        print(e)
        ax[0, 0].set_title("Figure-Eight Task")
        ax[0, 1].set_title("Straight-Away Task")
        fig.suptitle(f"Mean Error Magnitude ({physics_model})")
        plt.show()

if __name__ == "__main__":
    plot_summary_data()