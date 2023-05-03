# Investigating Deep Reinforcement Learning for Quadrotor Planning and Control



## Andrew Garrett, Ramya Muthukrishnan, Philip Sieg


For our ESE650 Final Project, we implemented a Model Predictive Path Intergral (MPPI) control scheme for Crazyflie drones in the gym-pybullet-drones simulation environment. (ADD MOTIVATION FOR MPPI) 

Our project can be broken up into the following sections:

- Data Generation: Using the pybullet-drones simualtor, we generated a large and expansive amount of trajectories. These trajectories were chosen to maximize state space coverage so that a model learning the system dynamics would be robust. 
- Dynamics Models: In order to evaluate our controller, we need an accurate model $x_{i+1}f(x_i,\ u_i)$ which predicts the next state given the current state ($x$) and a desired control ($u$, RPMs of four rotors). For our drone, the kinematics are rather trivial given the velocities, so we need to only learn the dynamics (acceleration). To calculate the accelerations, we have two different approaches. The first is an standard, analytical model for a quadcopter that finds the accelerations (linear and angular) via conservation of linear and angular momentum. The second approach is via a learned model, where we train a MLP on the data collected from the simulator. This second option allows us to more easily learn complex phenomena like drag or ground effect without the need to greatly rewrite the model. Instead, we can just retrain on new simulator data 
 
- MPPI Control (WRITE SOME MORE)


## Setup instructions


If you don't have anaconda, please download it from their website.  The build will probably take some time, so take a break while things are installing. :)

```
conda create -n quad_rl python=3.8
conda activate quad_rl
git clone https://github.com/andrew-garrett/gym-pybullet-drones.git
cd gym-pybullet-drones/
pip install -e .
cd ..
```


## Bootstrap Dataset Collection


If you have the above requirements installed but would like to record videos of trajectories, you must install `ffmpeg` in your conda
environment.

In general, the bootstrap module simulates a dataset of trajectories with options to visualize and or record flight data, including video.
The main functionalities are given below

### To collect the default trajectory dataset:
```
python bootstrap/boot_strap.py --tracking_config data
```

### To visualize the default trajectory dataset:
```
python bootstrap/boot_strap.py
```

### General Usage:
```
python bootstrap/boot_strap.py -h

usage: boot_strap.py [-h] [--task-battery {AGGRO,CONTROLLED,DEBUG,FULL}] [--tracking-config {default,debug,data,video_data,debug_video_data}]

Bootstrap Simulation Dataset Collection Script

optional arguments:
  -h, --help            show this help message and exit
  --task-battery {AGGRO,CONTROLLED,DEBUG,FULL}
                        task_battery.TaskBattery (default: DEBUG)
  --tracking-config {default,debug,data,video_data,debug_video_data}
                        controls what kind of visualization and whether or not we collect data (default: default)
```

The `--task-battery` argument controls the flavor of trajectory dataset to simulate.  `--task-battery FULL` uses multithreading to generate 
all of the datasets available in the `task_battery.TaskBattery` at one time.  It overrides any argument passed to `--tracking-config`
and uses the most lightweight procedure for collecting data.  This functionality can mainly be used for the fastest trajectory dataset generation.

The `--tracking-config` argument controls the level of visualization and volume of data collected for a given `task_battery.TaskBattery`.
Here is a brief description of each choice for the `--tracking-config` argument:
 - `--tracking-config default`: Visualizes the world and the quadrotor flight (No Trajectory Data Collection)
 - `--tracking-config debug`:  Visualizes the world, the quadrotor flight, and extra markers/lines for debugging (No Trajectory Data Collection)
 - `--tracking-config data`: Collects trajectory dataset (No Visualization or Video Recording)
 - `--tracking-config video_data`: Collects trajectory dataset and records videos visualizing the world and the quadrotor flight
 - `--tracking-config debug_video_data`: Collects trajectory dataset and records videos visualizing the world, the quadrotor flight, and extra markers/lines for debugging

 Note: Collecting Data take a few minutes, and visualization slows things down a good bit.  `--tracking-config debug_video_data` offers the most verbose data collection but is quite slow.  `--tracking-config data` or `--tracking-config default` are the two quickest procedures.  mp4s are created from images taken at each control timestep, which are muxxed such that the fps closely matches the control frequency.  Up to 8 quadrotors can be visualized in the simulator, but dataset collection is performed with somewhere between 27 and 64 drones per dataset.

 To change the physics model used by pybullet, change the control frequency, set postional offsets of the drones, and other simulation environment related parameters, consider changing values in [`./configs/tracking/tracking_config.json`](./configs/tracking/tracking_config.json).


## Dynamics Model Training


First, please use pip/conda to install/update torch, torchvision, and pytorch-lightning in your conda environment.  There may be mismatched
dependencies so be careful to choose the correct installs for your hardware and virtual environment.

### To create a torch dataset from a bootstrapped trajectory dataset:
```
python torch_dataset_gen.py --dataset_path path/to/dataset/root/dir
```

### General Usage
```
python torch_dataset_gen.py  -h

usage: torch_dataset_gen.py [-h] [--dataset_path] [--seed] [--val_split VAL_SPLIT] [--max_rpm MAX_RPM]

Pytorch dataset file generation script

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path        Root path of dataset (default: ./bootstrap/datasets/AGGRO_000)
  --seed                Seed for dataset splitting (default: 42)
  --val_split VAL_SPLIT
                        Fraction of dataset used for validation (default: 0.2)
  --max_rpm MAX_RPM     Maximum RPM value in dataset (default: 21702.6438)
```

Training hyperparameters are stored in [`./configs/training_config.json`](./configs/training_config.json), and can be easily configured for
experiments.

### To train a dynamics neural network on a torch dataset:
```
python train_dynamics_model.py
```

### To log training statistics to Weights and Biases (assuming you've installed it and have an API key):
```
python train_dynamics_model.py --sweep ./configs/sweep_config.json
```

The [`./configs/sweep_config.json`](./configs/sweep_config.json) can be used to test different parameters in a dedicated config file, and log their results to WandB.
This module is also compatible with Hyperparameter Sweeps on Weights and Biases (WandB).  [Here](./) is our current WandB
project with data from lots of bad models and a few very good ones.

### General Usage
```
python train_dynamics_model.py -h

usage: train_dynamics_model.py [-h] [--root ROOT] [--dataset DATASET] [--config CONFIG] [--epochs EPOCHS] [--sweep SWEEP]

Dynamics Neural Network Training Script

optional arguments:
  -h, --help         show this help message and exit
  --root ROOT        root where datasets are stored (default: ./bootstrap/datasets/)
  --dataset DATASET  name of the dataset (default: AGGRO_000)
  --config CONFIG    path_to_config_file (default: training_config.json)
  --epochs EPOCHS    Number of training epochs (default: 10)
  --sweep SWEEP      sweep config file (default: None)
```


# TODO

- [ ] Tune cost function to improve MPPI results 
- [ ] Debug MPPI yaw spinning out with initial orientation close to -2*pi
- [x] Implement PID as a baseline for MPPI controller
- [ ] Create learned models for PyBullet physics with more complicated phenomena (drag, downwash, etc)
- [ ] Add Controller specification to /configs/tracking/tracking_config.json
- [ ] Improve Trajectory Generation Scheme (smarter waypoint filtering, yaw trajectory generator)
- [ ] Prepare to Merge mppi branch into main

