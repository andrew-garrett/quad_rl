import random
import torch
from torch.distributions import Normal
import cupy as cp
import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt

# Hacky fix
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# PARAMETERS
SEED = 0
num_iters = 1000
T_horizon = 2.5
control_freq = 48
K_range = [i**3 for i in range(3, 12)]
T = int(T_horizon*control_freq)
control_space = 4 # rpms, for now
control_noise = 10. # rpm noise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set seeds
torch.manual_seed(SEED)
cp.random.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def torch_sampling():
    torch_rng = Normal(0.0, control_noise)
    torch_avg_runtime = np.zeros((num_iters, len(K_range)))
    for i in range(num_iters):
        for j, K in enumerate(K_range):
            t_i = time()
            torch_samples = torch_rng.sample((K, T, 4))
            t_f = time()
            torch_avg_runtime[i, j] = t_f - t_i
    
    return np.mean(torch_avg_runtime, axis=0)


def cupy_sampling():
    cp_avg_runtime = np.zeros((num_iters, len(K_range)))
    for i in range(num_iters):
        for j, K in enumerate(K_range):
            t_i = time()
            cp_samples = cp.random.normal(loc=0., scale=control_noise, size=(K, T, 4), dtype=np.float32)
            t_f = time()
            cp_avg_runtime[i, j] = t_f - t_i
    
    return np.mean(cp_avg_runtime, axis=0)


def numpy_sampling():
    np_rng = np.random.default_rng(seed=SEED)
    np_avg_runtime = np.zeros((num_iters, len(K_range)))
    for i in range(num_iters):
        for j, K in enumerate(K_range):
            t_i = time()
            np_samples = np_rng.normal(loc=0., scale=control_noise, size=(K, T))
            t_f = time()
            np_avg_runtime[i, j] = t_f - t_i
    
    return np.mean(np_avg_runtime, axis=0)


torch_avg_runtime = torch_sampling()
cp_avg_runtime = cupy_sampling()
np_avg_runtime = numpy_sampling()

samples_per_K = [T*K for K in K_range]

plt.plot(samples_per_K, 1.0/torch_avg_runtime, label="torch")
plt.plot(samples_per_K, 1.0/cp_avg_runtime, label="cupy")
plt.plot(samples_per_K, 1.0/np_avg_runtime, label="numpy")
plt.title("Random Sampling Frequencies")
plt.xlabel("Number of Samples")
plt.ylabel("Log Frequency (Hz)")
plt.yscale("log")
plt.legend()
plt.show()