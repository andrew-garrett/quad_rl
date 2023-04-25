import argparse
from copy import deepcopy
import json
import os, sys
sys.path.append("..\\quad_rl")
import numpy as np
import torch
from tqdm import tqdm
import wandb

"""
MPPI Tunable parameters:

Scalars
    - K: Number of sampled trajectories
    - T_HORIZON: Trajectory length in seconds
    - TEMPERATURE: Controls the density/spread around the mean of the control distribution
    - SYSTEM_NOISE: The covariance of control perturbation distribution
    - SYSTEM_BIAS: The mean of the control perturbation distribution

Vectors
    - Q: Diagonal State-Action Cost Covariance
        - Q_p, Q_q, Q_v, Q_w can be the scalar quantities we optimize

We will do a parameter sweep as follows:

Scalars:
    - K: Categorical(32, 64, 128, 512)
    - T_HORIZON: Uniform(min=0.5, max=2.5)
    - TEMPERATURE: Log_Uniform(min=0.05, max=5)
    - SYSTEM_NOISE: Int_Uniform(min=1, max=100)
    - SYSTEM_BIAS: Quantized_Normal(mu=0.0, sigma=500.0, q=10)
    - Q_p: Uniform(min=1.0, max=10.0)
    - Q_r: Uniform(min=1.0, max=50.0)

We will tune the model with the following procedure:

NUM_TRIALS = 3
NUM_TASKS = 3 (to start out, then we can add more)

while true:
    set_hyperparameters (wandb/ray)
    task_costs = zeros(NUM_TASKS)
    for task_num in range(NUM_TASKS):
        for trial_num in range(num_trials):
            costs[task_num] += run_task(task_num)
        
Tuning Strategies and Notes:

    Upon initial tuning with parameters similar to those above, I found that the total cost is dominated by the tracked trajectory term.
This could be due to many things.
First, we are scaling the rollout costs by the number of timesteps in the simulate() function.  This intuitively makes sense if we want
the tracked-trajectory-costs and rollout costs to be on the same scale (we compute T full T-length rollouts)

Although, this behavior makes some sense because of how we are currently computing the control cost:
    
    We started out by dividing the control cost by MAX_RPM**2, which also made sense because our system_noise, current control, and 
control perturbation are all considered in RPM.  When taking that term away, the control-cost term can explode.  Now that I think of
this, the normalizing factor should be MAX_RPM, not MAX_RPM**2.  We want to normalize each of the terms independently.  Regardless,

This results in models biasing toward low-noise.  Coupled with this is the fact that allowing a tuner to vary the time-horizon.
means that it will also minimize this quantity.  One thing we should do is just set T_horizon to 1.0.

In order to mitigate the dominanace of the control term in the cost function, I have a couple ideas:
1. Ensure that both the delta_x term (in the state-dependent cost) and the u_tm1


Evaluation:

Start at hover.  Set target positions to be 2 meters away in +/-x, +/-y, and the diagonals.  Set target velocity to be the distance over the time horizon.
At the start of a test, perform several iterations of MPPI as warmup (not rolling the controls) and measure the cost of the optimal rollout at the start.
Then simulate MPPI for T-horizon timesteps and measure the state-cost for the final timestep.

"""

from testMPPI_v2 import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        help="sweep config file",
        type=str
    )
    args = parser.parse_args()
    if args.sweep is not None:
        # Save the new sweep config
        with open(args.sweep, "r") as f:
            wand_config_dict = json.load(f)
        with open("./configs/sweep_config.json", "w") as f:
            json.dump(wand_config_dict, f, indent="\t", sort_keys=True)

        eval_dict = evaluate(sweep_config_path="./configs/sweep_config.json")
        for k, v in eval_dict.items():
            print(k, v)
        wandb.finish()
    else:
        eval_dict = evaluate() # sweep_config_path="./configs/mppi_config.json"
        for k, v in eval_dict.items():
            print(k, v)



# import mppi.cost_models as cost_models
# import mppi.dynamics_models as dynamics_models
# import mppi.MPPI_Node as MPPI_Node
# from mppi.testMPPI import simulate

# NUM_TRIALS = 1


# def evaluate(num_trials=NUM_TRIALS, sweep_config_path=None):

#     if sweep_config_path is None:
#         config_fpath = "./configs/mppi_config.json"
#     else:
#         # Sets the parameters of the MPPI config according to wandb sweep
#         config_fpath = sweep_config_path
#         logging_config = {
#             "reinit": True,
#             "project": "ESE650 Final Project",
#             "group": "Unit Step MPPI Tuning (Savgol Filter)"
#         }
#         wandb_run = wandb.init(**logging_config)
#     costs = {
#         "tracked_trajectory_cost": [],
#         "mean_optimal_trajectory_cost": [],
#         "sample_trajectories": []
#     }

#     # evaluate over a diverse dataset of trajectories
#     ##### Initial State
#     INIT_XYZS = np.array([0., 0., 1.,])
#     INIT_RPYS = np.array([0., 0., 0.])
#     INIT_STATE = np.hstack((INIT_XYZS, INIT_RPYS, np.zeros(6)))

#     grid_positions = np.indices((3, 3, 3))
#     grid_positions = grid_positions.reshape(3, -1).T
#     TARGET_STATES = np.zeros((grid_positions.shape[0], 12))
#     TARGET_STATES[:, :3] = grid_positions
#     TARGET_STATES = TARGET_STATES[TARGET_STATES[:, 2] == INIT_XYZS[2], :]

#     verbose_index = np.random.randint(0, TARGET_STATES.shape[0], 3)
#     with tqdm(total=num_trials*TARGET_STATES.shape[0], postfix=[""]) as tq:
#         for i, TARGET_STATE in enumerate(TARGET_STATES):
#             if not np.all(TARGET_STATE == INIT_STATE):
#                 for j in range(num_trials):
#                     # Get a fresh MPPI Config
#                     mppi_config = MPPI_Node.get_mppi_config(config_fpath)
#                     tracked_trajectory_cost, mean_optimal_trajectory_cost, gif_fps = simulate(mppi_config, deepcopy(INIT_STATE), deepcopy(TARGET_STATE), verbose=(i in verbose_index))
#                     costs["tracked_trajectory_cost"].append(tracked_trajectory_cost)
#                     costs["mean_optimal_trajectory_cost"].append(mean_optimal_trajectory_cost)
#                     if gif_fps is not None and sweep_config_path is not None:
#                         wandb.log({"sample_trajectory": wandb.Video("./trajectory.gif", fps=int(mppi_config.FREQUENCY))})
#                     tq.update()
#     costs["tracked_trajectory_cost_arr"] = [np.mean(cost_arr) for cost_arr in costs["tracked_trajectory_cost"]]
#     costs["tracked_trajectory_cost"] = np.mean(costs["tracked_trajectory_cost_arr"]) / np.linalg.norm(costs["tracked_trajectory_cost_arr"])
#     costs["mean_optimal_trajectory_cost_arr"] = [np.mean(cost_arr) for cost_arr in costs["mean_optimal_trajectory_cost"]]
#     costs["mean_optimal_trajectory_cost"] = np.mean(costs["mean_optimal_trajectory_cost_arr"]) / np.linalg.norm(costs["mean_optimal_trajectory_cost_arr"])
#     costs["total_cost"] = costs["tracked_trajectory_cost"]+costs["mean_optimal_trajectory_cost"]
#     if sweep_config_path is not None:
#         wandb.log(costs)
#     return costs


















# their_mppi_config = MPPI_Node.get_mppi_config(config_fpath)
# if their_mppi_config.METHOD.__name__ == "torch":
#     their_next_state = their_mppi_config.METHOD.asarray(INIT_STATE).to(device=their_mppi_config.DEVICE, dtype=their_mppi_config.DTYPE)
#     their_target_state = their_mppi_config.METHOD.asarray(TARGET_STATE).to(device=their_mppi_config.DEVICE, dtype=their_mppi_config.DTYPE)
# else:
#     their_next_state = their_mppi_config.METHOD.asarray(INIT_STATE, dtype=their_mppi_config.DTYPE)
#     their_target_state = their_mppi_config.METHOD.asarray(TARGET_STATE, dtype=their_mppi_config.DTYPE)
# their_mppi = MPPI(
#                     dynamics=dynamics_models.AnalyticalModel(their_mppi_config), 
#                     running_cost=cost_models.CostModel(their_mppi_config, TARGET_STATE), 
#                     nx=their_mppi_config.X_SPACE, 
#                     noise_sigma=torch.from_numpy(their_mppi_config.SYSTEM_NOISE).to(device=their_mppi_config.DEVICE, dtype=their_mppi_config.DTYPE), 
#                     num_samples=their_mppi_config.K,
#                     horizon=their_mppi_config.T,
#                     device=their_mppi_config.DEVICE,
#                     terminal_state_cost=None,
#                     lambda_=their_mppi_config.TEMPERATURE,
#                     noise_mu=None,
#                     u_min=torch.zeros(their_mppi_config.U_SPACE).to(device=their_mppi_config.DEVICE, dtype=their_mppi_config.DTYPE), 
#                     u_max=their_mppi_config.CF2X.MAX_RPM*torch.ones(their_mppi_config.U_SPACE).to(device=their_mppi_config.DEVICE, dtype=their_mppi_config.DTYPE),  
#                     u_init=their_mppi_config.CF2X.HOVER_RPM*torch.ones(their_mppi_config.U_SPACE).to(device=their_mppi_config.DEVICE, dtype=their_mppi_config.DTYPE), 
#                     U_init=their_mppi_config.CF2X.HOVER_RPM*torch.ones((their_mppi_config.T, their_mppi_config.U_SPACE)).to(device=their_mppi_config.DEVICE, dtype=their_mppi_config.DTYPE), 
#                     u_scale=1,
#                     u_per_command=1, 
#                     step_dependent_dynamics=False, 
#                     rollout_samples=1, 
#                     rollout_var_cost=0, 
#                     rollout_var_discount=0.95, 
#                     sample_null_action=False, 
#                     noise_abs_cost=True
#                 )
# their_mppi_test = MPPITest(their_mppi_config, their_mppi)

# from autotune import EvaluationResult, AutotuneMPPI, CMAESOpt
# # use the same nominal trajectory to start with for all the evaluations for fairness
# nominal_trajectory = their_mppi.U.clone()
# # parameters for our sample evaluation function - lots of choices for the evaluation function
# evaluate_running_cost = True
# num_refinement_steps = 10
# num_trajectories = 5

# init_state_torch = torch.from_numpy(INIT_STATE).to(device=mppi_config.DEVICE, dtype=mppi_config.DTYPE)
# def evaluate():
#     costs = []
#     rollouts = []
#     # we sample multiple trajectories for the same start to goal problem, but in your case you should consider
#     # evaluating over a diverse dataset of trajectories
#     for j in range(num_trajectories):
#         their_mppi.U = nominal_trajectory.clone()
#         # the nominal trajectory at the start will be different if the horizon's changed
#         their_mppi.change_horizon(their_mppi.T)
#         # usually MPPI will have its nominal trajectory warm-started from the previous iteration
#         # for a fair test of tuning we will reset its nominal trajectory to the same random one each time
#         # we manually warm it by refining it for some steps
#         for k in range(num_refinement_steps):
#             their_mppi.command(init_state_torch, shift_nominal_trajectory=False)

#         rollout = their_mppi.get_rollouts(init_state_torch)

#         this_cost = 0
#         rollout = rollout[0]
#         # here we evaluate on the rollout MPPI cost of the resulting trajectories
#         # alternative costs for tuning the parameters are possible, such as just considering terminal cost
#         if evaluate_running_cost:
#             for t in range(len(rollout) - 1):
#                 this_cost = this_cost + their_mppi.running_cost(rollout[t], their_mppi.U[t])
#         # this_cost = this_cost + env.terminal_cost(rollout, mppi.U)

#         rollouts.append(rollout)
#         costs.append(this_cost)
#     # can return None for rollouts if they do not need to be calculated
#     try:
#         costs = torch.stack(costs)
#     except:
#         costs = torch.tensor(costs)
#     return EvaluationResult(costs, torch.stack(rollouts))



# import autotune_global
# from ray import tune
# from ray.tune.search.hyperopt import HyperOptSearch
# from ray.tune.search.bayesopt import BayesOptSearch

# # be sure to close any figures before ray tune optimization or they will be duplicated
# # env.visualize = False
# plt.close('all')
# # choose from autotune.AutotuneMPPI.TUNABLE_PARAMS
# params_to_tune = ['sigma', 'mu', 'horizon', 'lambda']
# tuner = autotune_global.AutotuneMPPIGlobal(their_mppi, params_to_tune, evaluate_fn=evaluate,
#                                            optimizer=autotune_global.RayOptimizer(HyperOptSearch),
#                                            sigma_search_space=tune.loguniform(1e-4, 1e2),
#                                            mu_search_space=tune.uniform(-1, 1),
#                                            lambda_search_space=tune.loguniform(1e-5, 1e3),
#                                            horizon_search_space=tune.randint(10, 100)
#                                            )
# # ray tuners cannot be tuned iteratively, but you can specify how many iterations to tune for
# res = tuner.optimize_all(100)
# res = tuner.get_best_result()
# tuner.apply_parameters(res.params)

# # choose from autotune.AutotuneMPPI.TUNABLE_PARAMS
# params_to_tune = ['sigma', 'mu', 'horizon', 'lambda']
# # create a tuner with a CMA-ES optimizer
# tuner = AutotuneMPPI(their_mppi, params_to_tune, evaluate_fn=evaluate, optimizer=CMAESOpt(sigma=1.0))
# # tune parameters for a number of iterations
# iterations = 5
# for i in tqdm(list(range(iterations))):
#     # results of this optimization step are returned
#     res = tuner.optimize_step()
# # get best results and apply it to the controller
# # (by default the controller will take on the latest tuned parameter, which may not be best)
# res = tuner.get_best_result()
# tuner.apply_parameters(res.params)
# print(res)