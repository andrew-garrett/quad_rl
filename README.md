# Investigating Deep Reinnforcement Learning for Quadrotor Planning and Control

## Andrew Garrett, Ramya Muthukrishnan, Philip Sieg

For our ESE650 Final Project, we implemented a Model Predictive Path Intergral (MPPI) control scheme for Crazyflie drones in the gym-pybullet-drones simulation environment. (ADD MOTIVATION FOR MPPI) 

Our project can be broken up into the following sections:

- Data Generation: Using the pybullet-drones simualtor, we generated a large and expansive amount of trajectories. These trajectories were chosen to maximize state space coverage so that a model learning the system dynamics would be robust. 
- Dynamics Models: In order to evaluate our controller, we need an accurate model $x_{i+1}f(x_i,\ u_i)$ which predicts the next state given the current state ($x$) and a desired control ($u$, RPMs of four rotors). For our drone, the kinematics are rather trivial given the velocities, so we need to only learn the dynamics (acceleration). To calculate the accelerations, we have two different approaches. The first is an standard, analytical model for a quadcopter that finds the accelerations (linear and angular) via conservation of linear and angular momentum. The second approach is via a learned model, where we train a MLP on the data collected from the simulator. This second option allows us to more easily learn complex phenomena like drag or ground effect without the need to greatly rewrite the model. Instead, we can just retrain on new simulator data 
 
- MPPI Control (WRITE SOME MORE)

# Setup instructions

This will probably take some time, so take a break while things are installing.s :)

```
conda create -n quad_rl python=3.8
conda activate quad_rl
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/
pip install -e .
cd ..
```

TBD, but to record data videos, we may need to install ffmpeg.

If that becomes an issue, I'll create a docker image for this project.

# Example Usage

Below should generate a vizualization window of 3 quads flying along a helical path.  Then, a matplotlib window should pop up and show state and control data from all of the drones.

```
python gym-pybullet-drones/gym_pybullet_drones/examples/fly.py
```

To train an agent to hover:

```
python gym-pybullet-drones/experiments/learning/singleagent.py --cpu 2
```

To view training results, in either a separate terminal or after running the above:
```
tensorboard --logdir ./results
```

# Dynamics Model Training

First, please use pip to install torch, torchvision, and pytorch-lightning in your conda environment

Then, create a torch dataset by running:

```
python torch_dataset_gen.py --dataset_path path/to/dataset/root/dir
```

Then, after accordingly setting parameters in training_config.json, run training:

```
python train_dynamics_model.py
```


# Bootstrap Dataset Collection

If you have the above requirements installed, then you should be good to start collecting data.

Datasets are highly parameterizable, perhaps a bit too much so right now.  To generate the default debug dataset:

```
python bootstrap/boot_strap.py
```

Additional arguments can be provided as:

```
usage: boot_strap.py [-h] [--task-battery]        

Bootstrap Simulation Dataset Collection Script    

optional arguments:
  -h, --help       show this help message and exit
  --task-battery   task_battery.TaskBattery 
```
The options for `task_battery.TaskBattery` are in the bootstrap/task_battery.py, where there are options for a DEBUG dataset,
an AGGRO flight dataset, and a CONTROLLED flight dataset.  Further options for the simulation dataset are configured
in the configs/tracking_config.json.  These options control parameters such as the physics model, whether to display to the GUI,
and simulation speed parameters.




# TODO:

- Tune cost function to improve MPPI results 
- Implement PID as a baseline for MPPI controller 
- Create learned models for PyBullet physics with more complicated phenomena (drag, downwash, etc) 

